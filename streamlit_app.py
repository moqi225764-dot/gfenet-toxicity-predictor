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
    st.session_state.pop('prediction_payload',None)
    clear_analysis()

def clear_analysis():
    st.session_state.pop('fingerprint_analysis',None)

def encode_file(file):
    return {'name':file.name,'b64':base64.b64encode(file.getvalue()).decode('ascii')}

st.caption('GFENet  /  多任务分子预测（本地免指纹版）')
st.title('呼吸毒性预测工作台')
st.write('只上传分子 CSV，四类指纹由服务端用 PaDEL 自动生成，选择关注方向即可获得可下载的多任务预测结果。')
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
    st.success('已准备 1 个真实测试分子。点击“开始预测”即可体验自动生成指纹的流程。')
    st.caption('这是既有留出测试样例，不是新开展的外部验证。')
else:
    st.write('分子 CSV 只需包含 `smiles` 列。四类指纹会在预测时自动生成，无需上传。')
    uploads['csv']=st.file_uploader('分子列表 CSV',type=['csv'],key='molecules',on_change=clear_result)
    st.caption('自动生成使用 PaDEL-Descriptor v2.21 默认参数（Pubchem 881 / GraphOnly 1024 / AtomPairs2D 780 / Fingerprinter 1024），与训练指纹逐位一致。单个文件最大 5 MB。')

ready=bool(selected) and (mode=='使用真实测试样例' or uploads.get('csv'))
if st.button('开始预测',type='primary',disabled=not ready):
    clear_result()
    acquired=engine.INFERENCE_LOCK.acquire(blocking=False)
    if not acquired:
        st.warning('目前另一个预测正在计算，请稍后再次点击。')
    else:
        try:
            example=engine.example_payload() if mode=='使用真实测试样例' else {k:encode_file(v) for k,v in uploads.items()}
            payload={'csv':example['csv'],'tasks':selected}
            with st.spinner('正在检查文件、构建分子图并预测，请稍候…'):
                result=engine.run_predict(payload)
            st.session_state['prediction_result']=result
            st.session_state['prediction_payload']=payload
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f'预测未完成（{type(exc).__name__}）：{exc}')
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

    st.subheader('3 · 指纹影响分析')
    st.write('选择一个分子和预测方向，分别将一种指纹的输入全部置零，比较预测分数。其余输入保持不变。')
    st.caption('这是零基线屏蔽敏感性分析，不是门控权重或贡献百分比。指纹之间可能包含重叠信息；全零输入也可能偏离真实数据分布。')
    valid_rows=table.loc[table.status.eq('ok'),'source_row_zero_based'].astype(int).tolist()
    if valid_rows and st.session_state.get('prediction_payload'):
        smiles_by_row=table.set_index('source_row_zero_based')['smiles'].to_dict()
        row=st.selectbox('选择要分析的分子',valid_rows,
            format_func=lambda r:f'第 {r+1} 个分子 · {smiles_by_row[r]}',
            key='analysis_row',on_change=clear_analysis)
        task=st.selectbox('选择要分析的毒性方向',selected,
            format_func=lambda t:LABELS.get(t,t),key='analysis_task',on_change=clear_analysis)
        if st.button('分析指纹影响'):
            clear_analysis()
            acquired=engine.INFERENCE_LOCK.acquire(blocking=False)
            if not acquired:
                st.warning('目前另一个预测或分析正在计算，请稍后再次点击。')
            else:
                try:
                    with st.spinner('正在进行原始预测及四次指纹屏蔽实验…'):
                        analysis=engine.analyze_fingerprints(st.session_state['prediction_payload'],row,task)
                    st.session_state['fingerprint_analysis']=analysis
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error('分析未完成，请稍后重试或联系部署者检查日志。')
                finally:
                    engine.INFERENCE_LOCK.release()
        analysis=st.session_state.get('fingerprint_analysis')
        if analysis and analysis['row']==row and analysis['task']==task:
            st.metric('该分子的原始模型分数',f"{analysis['baseline']:.6f}")
            impact=pd.DataFrame(analysis['records'])
            delta='分数变化（原始−屏蔽）'
            st.bar_chart(impact.set_index('指纹')[[delta]],horizontal=True)
            st.caption('正值：屏蔽后分数下降；负值：屏蔽后分数上升。绝对值越大，模型对该次屏蔽越敏感。接近零不代表该指纹完全无用。四项变化不相加为总分。')
            st.dataframe(impact,width='stretch',hide_index=True,
                column_config={c:st.column_config.NumberColumn(format='%.6f') for c in ['原始分数','屏蔽后分数',delta]})
            st.download_button('下载指纹影响分析 CSV',impact.to_csv(index=False).encode('utf-8-sig'),
                f'fingerprint_impact_row_{row+1}.csv','text/csv')
    else:
        st.info('没有可供分析的有效分子，请先完成一次成功预测。')

with st.expander('模型说明与既有留出集指标'):
    st.write('该权重基于 2022 条有效记录训练，训练/验证/测试为 1416/303/303。按标准化分子分组划分，最佳权重和阈值只根据验证集选择。')
    st.caption('这不是外部验证，也不是骨架划分评估；标签缺失和共同阴性样本会影响对实际应用表现的判断。QM9 特征采用既有查表流程，未匹配的分子填零。')
    st.dataframe(pd.DataFrame(tasks).drop(columns=['index']).rename(columns={'name':'任务','threshold':'阈值','roc_auc':'留出 ROC-AUC','f1':'留出 F1','positive':'阳性数','labeled':'有标签数'}),width='stretch',hide_index=True)

