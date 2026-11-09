import json
from pathlib import Path
from ee6405.ingest.loaders import extract_week, load_coding_quiz_ocr, load_transcript, parse_clarifications, parse_mcq, load_notebook
from ee6405.rag.store import ChunkStore
from ee6405.types import Chunk

def test_week_and_clarification(tmp_path):
    p=tmp_path/'W03_Clarification.txt'; p.write_text('Week: 03\nDocument Type: Lecture Clarification\nRelated Material: W03_Class\nSlide Reference: Slide Page 2\nOriginal Clarification: answer here\n\nSlide Reference: Slide Page 4\nOriginal Clarification: second')
    docs=parse_clarifications(p); assert extract_week(p)=='W03'; assert len(docs)==2; assert docs[0].metadata['slide_page']=='Slide Page 2'

def test_mcq_answer_pair(tmp_path):
    m=tmp_path/'W01_Practice_MCQ.md'; a=tmp_path/'W01_Practice_Answers.txt'; m.write_text('# Q1: What?\n\n1. x'); a.write_text('# Week 1 #\nQ1: 3')
    d=parse_mcq(m,a); assert len(d)==1 and d[0].metadata['answer']=='3' and 'Answer: 3' in d[0].text

def test_notebook_markdown_code_pair(tmp_path):
    p=tmp_path/'W02_Code_New.ipynb'; p.write_text(json.dumps({'cells':[{'cell_type':'markdown','source':['Explain']},{'cell_type':'code','source':['x=1']},{'cell_type':'code','source':['y=2']}]}))
    d=load_notebook(p); assert len(d)==2 and '```python' in d[0].text and d[0].metadata['cell_idx']==0

def test_transcript_uses_corrected_text_and_time_windows(tmp_path):
    p=tmp_path/'W01_L01_Video.json'
    segments=[
        {'start': 0, 'end': 20, 'text': 'ChatGDP is wrong.', 'text_corrected': 'ChatGPT is correct.'},
        {'start': 20, 'end': 40, 'text': 'The model learns.', 'text_corrected': 'The model learns.'},
        {'start': 40, 'end': 60, 'text': 'It uses context.', 'text_corrected': 'It uses context.'},
        {'start': 60, 'end': 80, 'text': 'This is useful.', 'text_corrected': 'This is useful.'},
    ]
    p.write_text(json.dumps({'source':'W01/W01_L01_Video.mp4','week':'W01','lecture':'L01','segments':segments}))
    docs=load_transcript(p, chunk_size=20)
    assert len(docs)==2
    assert 'ChatGPT' in docs[0].text and 'ChatGDP' not in docs[0].text
    assert docs[0].modality=='transcript' and docs[0].metadata['lecture']=='L01'
    assert docs[0].metadata['start']==0 and docs[0].metadata['end']==60
    assert docs[0].metadata['source_video']=='W01/W01_L01_Video.mp4'
    assert docs[0].metadata['locator']=='t:00:00'

def test_coding_quiz_ocr_is_page_split(tmp_path):
    p=tmp_path/'Coding Quiz 3.md'
    p.write_text('## Page 1\n\nscaled_dot_product_attention(Q, K, V)\n\n## Page 2\n\nself_attn and cross_attn in DecoderLayer')
    docs=load_coding_quiz_ocr(p)
    assert len(docs)==2
    assert [d.metadata['page'] for d in docs]==[1,2]
    assert all(d.modality=='assessment_pdf' for d in docs)
    assert all(d.metadata['week'] is None and d.metadata['extraction']=='ocr' for d in docs)
    assert [d.metadata['locator'] for d in docs]==['page:1','page:2']

def test_week_filter(tmp_path):
    s=ChunkStore(tmp_path/'x.sqlite'); s.add_chunks([Chunk('a','d','transformer','s','slides',0,{'week':'01'},'page:1','01'),Chunk('b','d','transformer','s','slides',1,{'week':'02'},'page:2','02')]); assert [x[0].chunk_id if hasattr(x[0],'chunk_id') else x[0].chunk_id for x in s.search_bm25('transformer',week='02')]==['b']; s.close()

def test_search_excludes_mcq_modality(tmp_path):
    from ee6405.eval.run_eval import evaluate_mcq
    from ee6405.agent.llm import EchoLLM
    s=ChunkStore(tmp_path/'eval.sqlite')
    s.add_chunks([
        Chunk('q','d','What is TF IDF? 1. lexical 2. neural','q','mcq',0,{'week':'W03','qid':'1','answer':'1'},'q:1','W03'),
        Chunk('slide','d','TF IDF is a term weighting scheme','s','slides',0,{'week':'W03'},'page:1','W03'),
    ]); s.close()
    rows=evaluate_mcq(tmp_path/'eval.sqlite', retriever_kind='bm25', llm=EchoLLM())
    assert rows and all(x['source'] != 'q' for x in rows[0].retrieved)
