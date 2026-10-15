"""Portable CPU inference; no shared upload/result storage."""
import base64, io, json, time, threading
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger
from model import GFE
from features import smiletopyg
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent
CHECKPOINT=ROOT/'assets/gfenet_best.pt'
QM9_LOOKUP=ROOT/'assets/qm9_lookup.json'
METRICS_CSV=ROOT/'assets/test_metrics.csv'
EXAMPLE_DIR=ROOT/'example'
FINGERPRINTS=[('pubchem','PubChemFP（881 维）',881),('graph','GraphFP（1024 维）',1024),('apc2d','APC2D（780 维）',780),('fp1024','FP1024（1024 维）',1024)]
BATCH_SIZE=8
MAX_ROWS=50
MAX_ATOMS=80
_state={}
INFERENCE_LOCK=threading.Lock()

def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """按 utf-8 / gbk / latin1 依次尝试读取 CSV。"""
    last_err = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin1'):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
        except pd.errors.ParserError:
            raise
    raise ValueError(f'分子 CSV 编码无法识别：{last_err}')

def load_assets() -> None:
    """加载权重、任务名、阈值与 QM9 查表，进程内只执行一次。"""
    device = 'cpu'
    torch.set_num_threads(2)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = GFE(**ckpt['model_config'], device=device).to(device).eval()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

    lookup = {k: np.array(v, dtype=np.float32)
              for k, v in json.loads(QM9_LOOKUP.read_text(encoding='utf8')).items()}
    prep = ckpt['preprocessing']

    task_names = list(ckpt['task_names'])
    thresholds = [float(t) for t in ckpt['thresholds']]
    metrics = {}
    if METRICS_CSV.is_file():
        mt = pd.read_csv(METRICS_CSV)
        for _, row in mt.iterrows():
            metrics[str(row['task'])] = {
                'roc_auc': float(row['roc_auc']),
                'f1': float(row['f1']),
                'positive': int(row['positive']),
                'labeled': int(row['labeled']),
            }

    _state.update(
        device=device,
        model=model,
        task_names=task_names,
        thresholds=thresholds,
        metrics=metrics,
        qm9_lookup=lookup,
        electronic_mean=np.array(prep['electronic_mean'], dtype=np.float32),
        electronic_scale=np.array(prep['electronic_scale'], dtype=np.float32),
    )

def task_config() -> list:
    """返回全部呼吸毒性任务及其阈值与留出集指标。"""
    out = []
    for i, name in enumerate(_state['task_names']):
        m = _state['metrics'].get(name, {})
        out.append({
            'name': name,
            'index': i,
            'threshold': _state['thresholds'][i],
            'roc_auc': m.get('roc_auc'),
            'f1': m.get('f1'),
            'positive': m.get('positive'),
            'labeled': m.get('labeled'),
        })
    return out

def build_fingerprints(files: dict, n: int):
    """校验并归一化四类指纹，严格复刻训练脚本的处理方式。"""
    arrays, warnings = [], []
    for key, label, width in FINGERPRINTS:
        item = files.get(key)
        if not item or not item.get('b64'):
            raise ValueError(f'缺少 {label} 指纹文件')
        try:
            raw = base64.b64decode(item['b64'])
        except Exception:
            raise ValueError(f'{label} 文件内容无法解码')
        try:
            df = _read_csv_bytes(raw)
        except Exception as exc:
            raise ValueError(f'{label} 无法解析为 CSV：{exc}')

        if 'Name' in df.columns:
            names = df['Name'].astype(str).tolist()
            expect = [f'AUTOGEN_smi_{i + 1}' for i in range(n)]
            if names != expect:
                raise ValueError(f'{label} 的 Name 必须按 AUTOGEN_smi_1..{n} 排列，请核对分子与指纹行顺序。')
            df = df.drop(columns=['Name'])

        try:
            x = df.to_numpy(dtype=np.float32)
        except (ValueError, TypeError):
            raise ValueError(f'{label} 包含非数值内容，指纹必须全部为数值列')
        if x.shape != (n, width):
            raise ValueError(
                f'{label} 维度应为 {n} 行 × {width} 列，实际为 {x.shape[0]} 行 × {x.shape[1]} 列'
                f'（行数需与分子 CSV 一致）')
        if not np.isfinite(x).all():
            raise ValueError(f'{label} 存在 NaN 或 Inf 数值')

        sums = x.sum(1, keepdims=True)
        arrays.append(np.divide(x, sums, out=np.zeros_like(x), where=sums != 0))
    return arrays, warnings

def analyze_fingerprints(payload: dict, row: int, task: str) -> dict:
    """Single-row input-zero ablation; caller holds INFERENCE_LOCK.

    This is sensitivity to a zero baseline, not gate weights or causal attribution.
    Never mutate the original payload or shared model.
    """
    if task not in _state['task_names']:
        raise ValueError('未知预测任务')
    frame = _read_csv_bytes(base64.b64decode(payload['csv']['b64']))
    if not 0 <= row < len(frame) or len(frame) > MAX_ROWS:
        raise ValueError('分子行号超出范围')
    build_fingerprints(payload['fingerprints'], len(frame))
    def encode(frame):
        return {'b64': base64.b64encode(frame.to_csv(index=False).encode('utf-8')).decode('ascii')}
    single = {'csv': encode(frame.iloc[[row]]), 'fingerprints': {}, 'tasks': [task]}
    pieces = {}
    for key, _, _ in FINGERPRINTS:
        fp = _read_csv_bytes(base64.b64decode(payload['fingerprints'][key]['b64'])).iloc[[row]].copy()
        if 'Name' in fp:
            fp['Name'] = 'AUTOGEN_smi_1'
        pieces[key] = fp
        single['fingerprints'][key] = encode(fp)
    def score(data):
        result = run_predict(data)
        if result['summary']['ok'] != 1:
            raise ValueError('该分子无法预测，不能进行指纹分析')
        return float(pd.read_csv(io.BytesIO(result['csv_bytes']))[task + '_probability'].iloc[0])
    baseline = score(single)
    records = []
    for key, label, _ in FINGERPRINTS:
        masked = pieces[key].copy()
        cols = [c for c in masked if c != 'Name']
        masked.loc[:, cols] = 0
        altered = {**single, 'fingerprints': {**single['fingerprints'], key: encode(masked)}}
        value = score(altered)
        delta = baseline - value
        records.append({'分子序号': row + 1, '任务': task, '指纹': label,
                        '原始分数': baseline, '屏蔽后分数': value,
                        '分数变化（原始−屏蔽）': delta})
    return {'row': row, 'task': task, 'baseline': baseline, 'records': records,
            'method': 'input_zero_ablation'}


def run_predict(payload: dict) -> dict:
    """执行预测，返回可直接 JSON 化的结果结构。"""
    t0 = time.time()
    all_tasks = _state['task_names']

    # ---- 任务选择：模型始终输出全部任务，这里只决定展示/下载哪些列 ----
    selected = payload.get('tasks')
    if not selected:
        selected = list(all_tasks)
    selected = [t for t in selected if t in all_tasks]
    if not selected:
        raise ValueError('至少选择一个呼吸毒性任务')

    # ---- 分子 CSV ----
    csv_item = payload.get('csv') or {}
    if not csv_item.get('b64'):
        raise ValueError('缺少分子 CSV 文件')
    try:
        raw_csv = base64.b64decode(csv_item['b64'])
    except Exception:
        raise ValueError('分子 CSV 内容无法解码')
    try:
        df = _read_csv_bytes(raw_csv)
    except pd.errors.ParserError as exc:
        raise ValueError(f'分子 CSV 解析失败：{exc}')
    except Exception as exc:
        raise ValueError(f'分子 CSV 读取失败：{exc}')

    lower = {str(c).lower(): c for c in df.columns}
    smi_col = lower.get('smiles')
    if smi_col is None:
        raise ValueError('分子 CSV 必须包含 smiles 列')
    n = len(df)
    if n == 0:
        raise ValueError('分子 CSV 没有任何数据行')

    if n > MAX_ROWS:
        raise ValueError(f'云端版每次最多 {MAX_ROWS} 个分子，请分批上传。')

    fps, warnings = build_fingerprints(payload.get('fingerprints') or {}, n)

    # ---- 逐分子构图，记录无效原因 ----
    lookup = _state['qm9_lookup']
    mean = _state['electronic_mean']
    scale = _state['electronic_scale']
    graphs, rows, failures, valid_idx = [], [], [], []

    for i, smi in enumerate(df[smi_col].tolist()):
        smi = '' if smi is None else str(smi)
        reason = None
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError('SMILES 无法被 RDKit 解析')
            if mol.GetNumBonds() == 0:
                raise ValueError('分子不含任何化学键（单原子或离子），当前图流程不支持')
            if mol.GetNumAtoms() > MAX_ATOMS:
                raise ValueError(f'云端版单分子最多 {MAX_ATOMS} 个原子，请在本地处理较大分子')
            _, g = smiletopyg(smi, *(fp[i] for fp in fps), lookup)
            if g is None:
                raise ValueError('分子图构建失败')
            n_atoms, x, edges, ew, adj, dis, *tail = g
            fp1, fp2, fp3, fp4, q, e = tail
            if np.asarray(x).shape != (n_atoms, 40):
                raise ValueError(f'原子特征维度应为 {n_atoms}×40，实际为 {np.asarray(x).shape}')
            data = {'x': torch.tensor(x, dtype=torch.float32),
                    'edge_index': torch.tensor(edges, dtype=torch.long).T.contiguous(),
                    'edge_attr': torch.tensor(ew, dtype=torch.float32),
                    'electronic_desc': torch.tensor(((e - mean) / scale).reshape(1, -1),
                                                    dtype=torch.float32),
                    'qm9_features': torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, -1))}
            for j, fp in enumerate([fp1, fp2, fp3, fp4], 1):
                data[f'fp{j}'] = torch.tensor(np.asarray(fp, dtype=np.float32).reshape(1, -1))
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}' if not isinstance(exc, ValueError) else str(exc)
            failures.append({'row': i, 'smiles': smi, 'reason': reason})

        rows.append({'row': i, 'smiles': smi, 'status': reason or 'ok', 'valid': reason is None})
        if reason is None:
            valid_idx.append(i)
            graphs.append(Data(**data, adj=adj, dis=dis, leng=[n_atoms], row_id=i))

    # ---- 批量推理 ----
    probs = np.full((n, len(all_tasks)), np.nan, dtype=np.float64)
    if graphs:
        loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                p = _state['model'](batch).cpu().numpy()
                probs[batch.row_id.tolist(), :] = p
        if not np.isfinite(probs[valid_idx]).all():
            raise RuntimeError('模型输出出现非有限值，预测中止')

    # ---- 组装结果行 ----
    for r in rows:
        task_map = {}
        for j, name in enumerate(all_tasks):
            p = probs[r['row'], j]
            if not r['valid'] or not np.isfinite(p):
                task_map[name] = {'probability': None, 'prediction': None}
            else:
                task_map[name] = {'probability': float(p),
                                  'prediction': int(p >= _state['thresholds'][j])}
        r['predictions'] = task_map

    # ---- 汇总 ----
    ok_rows = [r for r in rows if r['valid']]
    per_task = {}
    for name in all_tasks:
        j = all_tasks.index(name)
        vals = [r['predictions'][name]['probability'] for r in ok_rows]
        vals = [v for v in vals if v is not None]
        pos = sum(r['predictions'][name]['prediction'] for r in ok_rows
                  if r['predictions'][name]['prediction'] is not None)
        per_task[name] = {
            'positive': pos,
            'negative': len(vals) - pos,
            'mean_probability': float(np.mean(vals)) if vals else None,
            'threshold': _state['thresholds'][j],
        }

    # ---- 落盘下载用的 CSV（含原始列 + 状态 + 所选任务） ----
    out = pd.DataFrame({'source_row_zero_based': np.arange(n)})
    drop = [smi_col] if smi_col != 'smiles' else []
    for c in df.columns:
        if c in drop or str(c).startswith('source_row_zero_based'):
            continue
        out[str(c)] = df[c].values
    if 'smiles' not in out.columns:
        out['smiles'] = df[smi_col].values
    out['status'] = [r['status'] for r in rows]
    for name in selected:
        out[f'{name}_probability'] = [r['predictions'][name]['probability'] for r in rows]
        out[f'{name}_prediction'] = [r['predictions'][name]['prediction'] for r in rows]

    download_name = (Path(str(csv_item.get('name') or 'predictions.csv')).stem
                     or 'predictions') + '_呼吸毒性预测.csv'
    buf = io.StringIO()
    out.to_csv(buf, index=False)

    return {
        'csv_bytes': buf.getvalue().encode('utf-8-sig'),
        'download_name': download_name,
        'device': _state['device'],
        'selected_tasks': selected,
        'tasks': task_config(),
        'rows': rows,
        'failures': failures,
        'per_task': per_task,
        'warnings': warnings,
        'summary': {'total': n, 'ok': len(ok_rows), 'failed': len(failures)},
        'elapsed': round(time.time() - t0, 2),
    }

def example_payload() -> dict:
    """把 example 目录下的官方示例打包成上传结构，供界面一键载入。"""
    mapping = [('csv', EXAMPLE_DIR / 'input.csv')]
    mapping += [(key, EXAMPLE_DIR / f'{key}.csv') for key, _, _ in FINGERPRINTS]
    out = {}
    for key, path in mapping:
        if not path.is_file():
            continue
        out[key] = {'name': path.name,
                    'b64': base64.b64encode(path.read_bytes()).decode('ascii')}
    return out
