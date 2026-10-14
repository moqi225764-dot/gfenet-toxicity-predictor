"""Public Streamlit interface for the existing GFENet checkpoint."""
import base64
import io
import zipfile
import pandas as pd
import streamlit as st
import engine

st.set_page_config(page_title='GFENet · 呼吸毒性预测', page_icon='🧬', layout='wide')
st.markdown('''<style>
.stApp {background: #f5f8fc;}
.block-container {max-width: 1180px; padding-top: 2rem;}
h1, h2, h3 {color: #142e46;}
[data-testid="stMetric"] {background: white; border: 1px solid #dce5ee; border-radius: 12px; padding: 15px;}
</style>''', unsafe_allow_html=True)

LABELS={
 'asthma&wheezing':'哮喘与喘息', 'bronchitis':'支气管炎', 'pneumonia':'肺炎',
 'bronchospasm':'支气管痉挛','embolism':'栓塞','pulmonary oedema':'肺水肿',
 'pulmonary fibrosis':'肺纤维化','pulmonary embolism':'肺栓塞',
 'Respiratory tract infection':'呼吸道感染','Respiratory sensitizer':'呼吸道致敏'}

@st.cache_resource
def initialize():
    engine.load_assets()
    return engine.task_config()

def clear_result():
    st.session_state.pop('prediction_result',None)

def encode_file(file):
    return {'name':file.name,'b64':base64.b64encode(file.getvalue()).decode('ascii')}

st.caption('GFENet  /  多任务分子预测')
st.title('呼吸毒性预测工作台')
st.write('上传分子与配套指纹，选择关注方向，获得可下载的多任务预测结果。')
st.info('研究用途：输出是模型分数和阈值分类，不是人体患病概率，也不能证明分子安全。模型尚未经过外部数据验证。')
try:
    with st.spinner('正在加载模型…'):
        tasks=initialize()
except Exception:
    st.error('模型资源加载失败，请联系部署者检查 assets 文件与依赖安装日志。')
    st.stop()
names=[t['name'] for t in tasks]

with st.sidebar:
    st.subheader('关注的预测方向')
    selected=st.multiselect('选择一个或多个任务',names,default=names,
        format_func=lambda x:LABELS.get(x,x),key='selected_tasks',on_change=clear_result)
    st.caption('模型计算全部任务；所选方向决定展示和下载的结果列。')
    st.divider()
    st.write(f'每批最多 **{engine.MAX_ROWS} 个分子**')
    st.write(f'每个分子最多 **{engine.MAX_ATOMS} 个原子**')
    st.caption('这是云端演示版的计算限制，较大任务请使用本地版本。')
    if st.button('清除预测结果'):
        clear_result()

st.subheader('1 · 准备输入')
mode=st.radio('输入方式',['使用真实测试样例','上传自己的文件'],horizontal=True,on_change=clear_result)
uploads={}
if mode=='使用真实测试样例':
    st.success('已准备 1 个真实测试分子及配套指纹。点击“开始预测”即可体验。')
    st.caption('这是既有留出测试样例，不是新开展的外部验证。')
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for filename in ['input.csv','pubchem.csv','graph.csv','apc2d.csv','fp1024.csv']:
            z.write(engine.EXAMPLE_DIR/filename,filename)
    st.download_button('下载示例输入文件',archive.getvalue(),'gfenet_example.zip','application/zip')
else:
    st.write('分子 CSV 需有 `smiles` 列。四类指纹必须与分子逐行对应，算法和参数必须与训练时一致。')
    uploads['csv']=st.file_uploader('分子列表 CSV',type=['csv'],key='molecules',on_change=clear_result)
    columns=st.columns(2)
    for i,(key,label,_) in enumerate(engine.FINGERPRINTS):
        with columns[i%2]:
            uploads[key]=st.file_uploader(label,type=['csv'],key=key,on_change=clear_result)
    st.caption('指纹若含 Name 列，编号须为 AUTOGEN_smi_1 起的连续编号。单个文件最大 5 MB。当前不支持仅凭 SMILES 自动生成四类指纹。')

ready=bool(selected) and (mode=='使用真实测试样例' or all(uploads.values()))
if st.button('开始预测',type='primary',disabled=not ready):
    clear_result()
    acquired=engine.INFERENCE_LOCK.acquire(blocking=False)
    if not acquired:
        st.warning('目前另一个预测正在计算，请稍后再次点击。')
    else:
        try:
            example=engine.example_payload() if mode=='使用真实测试样例' else {k:encode_file(v) for k,v in uploads.items()}
            payload={'csv':example['csv'],'fingerprints':{k:example[k] for k,_,_ in engine.FINGERPRINTS},'tasks':selected}
            with st.spinner('正在检查文件、构建分子图并预测，请稍候…'):
                result=engine.run_predict(payload)
            st.session_state['prediction_result']=result
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error('预测未完成。请检查输入文件，或联系部署者查看服务日志。')
        finally:
            engine.INFERENCE_LOCK.release()

result=st.session_state.get('prediction_result')
if result:
    st.subheader('2 · 预测结果')
    a,b,c=st.columns(3)
    a.metric('输入分子',result['summary']['total'])
    b.metric('预测成功',result['summary']['ok'])
    c.metric('无法预测',result['summary']['failed'])
    st.caption(f"用时 {result['elapsed']:.2f} 秒 · 阳性/阴性判定采用训练时保存在权重中的各任务阈值。")
    for warning in result['warnings']:st.warning(warning)
    table=pd.read_csv(io.BytesIO(result['csv_bytes']))
    st.dataframe(table,width='stretch',hide_index=True)
    st.download_button('下载预测结果 CSV',result['csv_bytes'],result['download_name'],'text/csv')
    if result['failures']:
        with st.expander('查看无法预测的记录'):
            st.dataframe(pd.DataFrame(result['failures']),width='stretch',hide_index=True)
    st.caption('结果只存放在当前会话中；清除结果、刷新连接或应用重启后可能丢失，请及时下载。')

with st.expander('模型说明与既有留出集指标'):
    st.write('该权重基于 2022 条有效记录训练，训练/验证/测试为 1416/303/303。按标准化分子分组划分，最佳权重和阈值只根据验证集选择。')
    st.caption('这不是外部验证，也不是骨架划分评估；标签缺失和共同阴性样本会影响对实际应用表现的判断。QM9 特征采用既有查表流程，未匹配的分子填零。')
    st.dataframe(pd.DataFrame(tasks).drop(columns=['index']).rename(columns={'name':'任务','threshold':'阈值','roc_auc':'留出 ROC-AUC','f1':'留出 F1','positive':'阳性数','labeled':'有标签数'}),width='stretch',hide_index=True)

