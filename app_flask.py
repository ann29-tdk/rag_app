"""
app_flask.py
============
Generalised RAG Web Application.

The Anthropic API call happens SERVER-SIDE — no browser key injection needed.
Works anywhere: local machine, server, VM.

Run:  python app_flask.py
Open: http://localhost:5000
Enter your Anthropic API key in the sidebar once, then ask questions.
"""

import os
import sys
import json
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, str(Path(__file__).parent))
from rag.engine import RAGEngine, _is_meta_query, _extract_author_lines

app    = Flask(__name__)
engine = RAGEngine()

# API key in server memory (set via sidebar or ANTHROPIC_API_KEY env var)
_api_key = os.environ.get("ANTHROPIC_API_KEY", "")


def _call_claude(question: str, chunks: list, api_key: str) -> str:
    """Call Anthropic API server-side. Returns coherent English answer."""
    context = "\n\n---\n\n".join(chunks[:8])
    system  = (
        "You are a precise research assistant. "
        "Answer the question using ONLY the document context provided. "
        "Be factual and specific — use exact figures, names, and terms from the context. "
        "Write in clear, well-structured English. "
        "If the answer has multiple parts, use a short numbered list. "
        "If the context does not contain enough information, say so clearly. "
        "Do NOT add information that is not in the context."
    )
    payload = json.dumps({
        "model"    : "claude-haiku-4-5-20251001",
        "max_tokens": 700,
        "system"   : system,
        "messages" : [{"role": "user",
                        "content": f"Document context:\n\n{context}\n\n---\n\nQuestion: {question}"}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {"Content-Type": "application/json",
                   "x-api-key": api_key,
                   "anthropic-version": "2023-06-01"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["content"][0]["text"]


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Research Assistant</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0a0c10;--surface:#111318;--surface2:#181b22;--border:#2a2d36;
    --accent:#4f9cf9;--accent2:#7c6af7;--green:#3ecf8e;--yellow:#f59e0b;
    --red:#ef4444;--text:#e2e8f0;--muted:#64748b;
    --mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;
  }
  body{background:var(--bg);color:var(--text);font-family:var(--sans);height:100vh;
       display:grid;grid-template-columns:300px 1fr;grid-template-rows:56px 1fr;overflow:hidden}
  header{grid-column:1/-1;background:var(--surface);border-bottom:1px solid var(--border);
         display:flex;align-items:center;padding:0 24px;gap:12px}
  .logo{font-size:22px;font-weight:800;letter-spacing:-0.5px}
  .logo span{color:var(--accent)}
  .tag{font-family:var(--mono);font-size:11px;background:var(--accent2);color:#fff;
       padding:2px 8px;border-radius:4px;opacity:.8}
  .stats-bar{margin-left:auto;display:flex;gap:20px;font-family:var(--mono);font-size:12px;color:var(--muted)}
  .stat-val{color:var(--green);font-weight:600}
  aside{background:var(--surface);border-right:1px solid var(--border);
        display:flex;flex-direction:column;padding:16px;gap:16px;overflow-y:auto}
  .section-label{font-family:var(--mono);font-size:10px;letter-spacing:1.5px;
                  color:var(--muted);text-transform:uppercase;margin-bottom:4px}
  .key-wrap{display:flex;gap:6px}
  .key-input{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:8px;
              color:var(--text);font-family:var(--mono);font-size:12px;padding:8px 10px;outline:none}
  .key-input:focus{border-color:var(--accent)}
  .key-input::placeholder{color:var(--muted)}
  .key-status{font-family:var(--mono);font-size:10px;margin-top:4px;padding:4px 8px;border-radius:4px}
  .key-ok  {background:rgba(62,207,142,.1);color:var(--green);border:1px solid rgba(62,207,142,.2)}
  .key-err {background:rgba(239,68,68,.1); color:var(--red);  border:1px solid rgba(239,68,68,.2)}
  .key-none{background:rgba(245,158,11,.1);color:var(--yellow);border:1px solid rgba(245,158,11,.2)}
  .upload-zone{border:1.5px dashed var(--border);border-radius:10px;padding:18px;
               text-align:center;cursor:pointer;transition:all .2s;position:relative}
  .upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:rgba(79,156,249,.05)}
  .upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%}
  .upload-icon{font-size:28px;margin-bottom:6px}
  .upload-text{font-size:13px;color:var(--muted)}
  .upload-text strong{color:var(--text);display:block;margin-bottom:2px}
  .btn{width:100%;padding:10px 16px;border-radius:8px;border:none;cursor:pointer;
       font-family:var(--sans);font-weight:600;font-size:13px;transition:all .15s}
  .btn-primary{background:var(--accent);color:#fff}
  .btn-primary:hover{background:#3b82f6;transform:translateY(-1px)}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none}
  .btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border);
             font-size:12px;padding:7px 12px}
  .btn-ghost:hover{color:var(--text);border-color:var(--accent)}
  .btn-danger{background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.2)}
  .btn-danger:hover{background:rgba(239,68,68,.2)}
  .btn-sm{width:auto;padding:6px 12px;font-size:12px}
  .doc-list{display:flex;flex-direction:column;gap:6px}
  .doc-item{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
            padding:10px 12px;display:flex;align-items:flex-start;gap:10px}
  .doc-icon{font-size:18px;flex-shrink:0;margin-top:1px}
  .doc-info{flex:1;min-width:0}
  .doc-name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .doc-meta{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:2px}
  .doc-del{background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;
            padding:0 4px;flex-shrink:0}
  .doc-del:hover{color:var(--red)}
  select{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);
         padding:8px 10px;border-radius:8px;font-family:var(--sans);font-size:13px;appearance:none}
  .action-row{display:flex;gap:8px}
  .action-row .btn{flex:1}
  .empty-docs{text-align:center;padding:16px;color:var(--muted);font-size:12px;font-family:var(--mono)}
  .progress-bar{height:3px;background:var(--border);border-radius:2px;overflow:hidden;display:none}
  .progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
                  width:0%;transition:width .3s;animation:shimmer 1.5s infinite}
  @keyframes shimmer{0%{opacity:1}50%{opacity:.6}100%{opacity:1}}
  main{display:flex;flex-direction:column;overflow:hidden}
  .chat-area{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;
              gap:20px;scroll-behavior:smooth}
  .chat-area::-webkit-scrollbar{width:6px}
  .chat-area::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  .welcome{max-width:600px;margin:auto;text-align:center;padding:40px 20px}
  .welcome-icon{font-size:56px;margin-bottom:16px}
  .welcome h2{font-size:28px;font-weight:800;margin-bottom:8px}
  .welcome p{color:var(--muted);font-size:14px;line-height:1.7}
  .example-qs{display:flex;flex-direction:column;gap:8px;margin-top:24px;text-align:left}
  .example-q{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
              padding:10px 14px;font-size:13px;color:var(--muted);cursor:pointer;transition:all .15s}
  .example-q:hover{border-color:var(--accent);color:var(--text)}
  .example-q::before{content:'→ ';color:var(--accent);font-weight:700}
  .msg{display:flex;gap:12px;max-width:100%;animation:fadeIn .3s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .msg.user{flex-direction:row-reverse}
  .avatar{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;
           justify-content:center;font-size:16px;flex-shrink:0}
  .avatar.user-av{background:linear-gradient(135deg,var(--accent),var(--accent2))}
  .avatar.ai-av{background:linear-gradient(135deg,var(--green),#059669)}
  .bubble{max-width:calc(100% - 60px);background:var(--surface2);
           border:1px solid var(--border);border-radius:12px;padding:14px 16px}
  .msg.user .bubble{background:rgba(79,156,249,.08);border-color:rgba(79,156,249,.2)}
  .bubble-text{font-size:14px;line-height:1.8;white-space:pre-wrap}
  .sources{margin-top:14px}
  .sources-label{font-family:var(--mono);font-size:10px;letter-spacing:1px;
                  color:var(--muted);text-transform:uppercase;margin-bottom:8px}
  .source-card{background:var(--surface);border:1px solid var(--border);
               border-radius:8px;margin-bottom:6px;overflow:hidden}
  .source-header{display:flex;align-items:center;gap:8px;padding:8px 12px;
                  cursor:pointer;user-select:none}
  .source-header:hover{background:var(--surface2)}
  .source-badge{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:4px;font-weight:600}
  .score-high{background:rgba(62,207,142,.15);color:var(--green)}
  .score-mid {background:rgba(245,158,11,.15); color:var(--yellow)}
  .score-low {background:rgba(100,116,139,.15);color:var(--muted)}
  .source-name{font-size:12px;font-weight:600;flex:1}
  .source-page{font-family:var(--mono);font-size:11px;color:var(--muted)}
  .chevron{color:var(--muted);font-size:12px;transition:transform .2s}
  .chevron.open{transform:rotate(90deg)}
  .source-body{display:none;padding:10px 12px;border-top:1px solid var(--border);
               font-size:12px;color:var(--muted);line-height:1.6;font-family:var(--mono)}
  .source-body.open{display:block}
  .thinking .bubble-text{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
  .dots span{display:inline-block;width:6px;height:6px;background:var(--accent);
              border-radius:50%;animation:bounce 1.2s infinite}
  .dots span:nth-child(2){animation-delay:.2s}
  .dots span:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-6px);opacity:1}}
  .input-bar{border-top:1px solid var(--border);padding:16px 24px;
              display:flex;gap:10px;background:var(--surface)}
  .input-wrap{flex:1;position:relative}
  textarea{width:100%;background:var(--surface2);border:1px solid var(--border);
            border-radius:10px;color:var(--text);font-family:var(--sans);font-size:14px;
            padding:12px 16px;resize:none;outline:none;transition:border-color .2s;
            min-height:48px;max-height:120px;line-height:1.5}
  textarea:focus{border-color:var(--accent)}
  textarea::placeholder{color:var(--muted)}
  .send-btn{width:48px;height:48px;background:var(--accent);border:none;border-radius:10px;
             cursor:pointer;display:flex;align-items:center;justify-content:center;
             font-size:18px;transition:all .15s;flex-shrink:0}
  .send-btn:hover{background:#3b82f6;transform:translateY(-1px)}
  .send-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
  .toast{position:fixed;bottom:80px;left:50%;transform:translateX(-50%);
          background:var(--surface2);border:1px solid var(--border);border-radius:8px;
          padding:10px 18px;font-size:13px;z-index:999;display:none;animation:fadeIn .3s}
  .toast.success{border-color:var(--green);color:var(--green)}
  .toast.error  {border-color:var(--red);  color:var(--red)}
  .toast.info   {border-color:var(--accent);color:var(--accent)}
  aside::-webkit-scrollbar{width:4px}
  aside::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>

<header>
  <div class="logo">🔍 <span>RAG</span>Assistant</div>
  <span class="tag">Any Document · Any Question</span>
  <div class="stats-bar">
    <span>Chunks: <span class="stat-val" id="hdr-chunks">0</span></span>
    <span>Docs: <span class="stat-val" id="hdr-docs">0</span></span>
    <span>Turns: <span class="stat-val" id="hdr-turns">0</span></span>
  </div>
</header>

<aside>
  <div>
    <div class="section-label">🔑 Anthropic API Key</div>
    <div class="key-wrap">
      <input type="password" id="api-key-input" class="key-input"
             placeholder="sk-ant-..." autocomplete="off">
      <button class="btn btn-primary btn-sm" onclick="saveKey()">Save</button>
    </div>
    <div id="key-status" class="key-status key-none">No key set</div>
    <div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:4px">
      Stored in server memory only. Never saved to disk.
    </div>
  </div>

  <div>
    <div class="section-label">📁 Upload Documents</div>
    <div class="upload-zone" id="dropzone">
      <input type="file" id="file-input" accept=".pdf,.txt,.docx" multiple>
      <div class="upload-icon">📄</div>
      <div class="upload-text">
        <strong>Drop files here or click</strong>
        PDF · DOCX · TXT — any topic
      </div>
    </div>
    <div class="progress-bar" id="progress-bar" style="margin-top:8px">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
  </div>

  <div>
    <div class="section-label">📚 Indexed Documents</div>
    <div class="doc-list" id="doc-list">
      <div class="empty-docs">No documents yet</div>
    </div>
  </div>

  <div>
    <div class="section-label">🔎 Search Scope</div>
    <select id="scope-select">
      <option value="">All Documents</option>
    </select>
  </div>

  <div>
    <div class="section-label">⚙️ Actions</div>
    <div class="action-row">
      <button class="btn btn-ghost" onclick="clearHistory()">🧹 Clear Chat</button>
      <button class="btn btn-ghost btn-danger" onclick="resetAll()">♻️ Reset</button>
    </div>
  </div>
</aside>

<main>
  <div class="chat-area" id="chat-area">
    <div class="welcome" id="welcome">
      <div class="welcome-icon">🔬</div>
      <h2>Document Q&amp;A Assistant</h2>
      <p>
        Upload any document — research papers, history books, reports,
        textbooks — then ask questions in plain English.<br><br>
        Enter your Anthropic API key in the sidebar, then get precise,
        well-structured answers with exact source citations.
      </p>
      <div class="example-qs">
        <div class="example-q" onclick="setQ('What are the main findings of this document?')">
          What are the main findings of this document?
        </div>
        <div class="example-q" onclick="setQ('Who are the authors and what institution are they from?')">
          Who are the authors and what institution are they from?
        </div>
        <div class="example-q" onclick="setQ('What methodology was used in this research?')">
          What methodology was used in this research?
        </div>
        <div class="example-q" onclick="setQ('Summarise the key conclusions.')">
          Summarise the key conclusions.
        </div>
      </div>
    </div>
  </div>

  <div class="input-bar">
    <div class="input-wrap">
      <textarea id="q-input" rows="1"
        placeholder="Ask anything about your document…"
        onkeydown="handleKey(event)"
        oninput="autoResize(this)"></textarea>
    </div>
    <button class="send-btn" id="send-btn" onclick="sendQuestion()">➤</button>
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
let isLoading = false;
let turnCount = 0;

window.addEventListener('load', () => {
  fetch('/key_status').then(r=>r.json()).then(d=>updateKeyStatus(d.set));
  fetch('/auto_index',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.success){showToast(`✅ Auto-indexed: ${d.filename} (${d.chunks} chunks)`,'success');
      refreshStats();refreshDocs();}
  }).catch(()=>{});
  refreshStats();refreshDocs();
});

async function saveKey(){
  const key=document.getElementById('api-key-input').value.trim();
  if(!key){showToast('Please enter your API key','error');return;}
  const r=await fetch('/set_key',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key})});
  const d=await r.json();
  updateKeyStatus(d.valid);
  if(d.valid){showToast('✅ API key saved — ready to answer questions','success');
    document.getElementById('api-key-input').value='';}
  else showToast('❌ '+(d.error||'Invalid key'),'error');
}

function updateKeyStatus(isSet){
  const el=document.getElementById('key-status');
  if(isSet){el.textContent='✓ Key set — ready';el.className='key-status key-ok';}
  else{el.textContent='⚠ No key — enter key above';el.className='key-status key-none';}
}

const dropzone=document.getElementById('dropzone');
const fileInput=document.getElementById('file-input');
dropzone.addEventListener('dragover',e=>{e.preventDefault();dropzone.classList.add('drag')});
dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('drag'));
dropzone.addEventListener('drop',e=>{e.preventDefault();dropzone.classList.remove('drag');uploadFiles(e.dataTransfer.files)});
fileInput.addEventListener('change',()=>uploadFiles(fileInput.files));

async function uploadFiles(files){
  if(!files.length)return;
  const bar=document.getElementById('progress-bar');
  const fill=document.getElementById('progress-fill');
  bar.style.display='block';fill.style.width='10%';
  for(let i=0;i<files.length;i++){
    const fd=new FormData();fd.append('file',files[i]);
    fill.style.width=`${20+(i/files.length)*70}%`;
    try{const r=await fetch('/upload',{method:'POST',body:fd});
        const d=await r.json();
        if(d.success)showToast(`✅ Indexed: ${d.filename} (${d.chunks} chunks, ${d.pages} pages)`,'success');
        else showToast(`❌ ${d.filename}: ${d.error}`,'error');
    }catch(e){showToast('❌ Upload failed: '+e.message,'error');}
  }
  fill.style.width='100%';
  setTimeout(()=>{bar.style.display='none';fill.style.width='0%'},600);
  fileInput.value='';refreshStats();refreshDocs();
}

function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendQuestion();}}
function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';}
function setQ(text){const inp=document.getElementById('q-input');inp.value=text;inp.focus();autoResize(inp);}

async function sendQuestion(){
  if(isLoading)return;
  const inp=document.getElementById('q-input');
  const question=inp.value.trim();
  if(!question)return;
  const scope=document.getElementById('scope-select').value;
  inp.value='';inp.style.height='auto';
  isLoading=true;document.getElementById('send-btn').disabled=true;
  const welcome=document.getElementById('welcome');
  if(welcome)welcome.style.display='none';
  appendMsg('user',question);
  const thinkId=appendThinking();
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question,source_filter:scope||null})});
    const d=await r.json();
    removeThinking(thinkId);
    if(d.error)appendMsg('assistant','❌ '+d.error);
    else{appendAnswer(d);turnCount++;document.getElementById('hdr-turns').textContent=turnCount;}
    refreshStats();
  }catch(e){
    removeThinking(thinkId);
    appendMsg('assistant','❌ Network error: '+e.message);
  }
  isLoading=false;document.getElementById('send-btn').disabled=false;
}

function appendMsg(role,text){
  const area=document.getElementById('chat-area');
  const div=document.createElement('div');
  div.className=`msg ${role}`;
  const isUser=role==='user';
  div.innerHTML=`<div class="avatar ${isUser?'user-av':'ai-av'}">${isUser?'👤':'🤖'}</div>
    <div class="bubble"><div class="bubble-text">${escHtml(text)}</div></div>`;
  area.appendChild(div);area.scrollTop=area.scrollHeight;
}

function appendThinking(){
  const area=document.getElementById('chat-area');
  const id='think-'+Date.now();
  const div=document.createElement('div');
  div.id=id;div.className='msg thinking';
  div.innerHTML=`<div class="avatar ai-av">🤖</div>
    <div class="bubble"><div class="bubble-text">Searching &amp; generating answer
      <div class="dots"><span></span><span></span><span></span></div>
    </div></div>`;
  area.appendChild(div);area.scrollTop=area.scrollHeight;return id;
}

function removeThinking(id){const el=document.getElementById(id);if(el)el.remove();}

function appendAnswer(data){
  const area=document.getElementById('chat-area');
  const div=document.createElement('div');div.className='msg';
  let sourcesHtml='';
  if(data.sources&&data.sources.length){
    const cards=data.sources.map(s=>{
      const pct=s.score_pct||0;
      const cls=pct>=60?'score-high':pct>=35?'score-mid':'score-low';
      return `<div class="source-card">
        <div class="source-header" onclick="toggleSource(this)">
          <span class="source-badge ${cls}">Match ${pct}%</span>
          <span class="source-name">📄 ${escHtml(s.source)}</span>
          ${s.page?`<span class="source-page">Page ${s.page}</span>`:''}
          <span class="chevron">›</span>
        </div>
        <div class="source-body">${escHtml(s.excerpt||'')}</div>
      </div>`;
    }).join('');
    sourcesHtml=`<div class="sources">
      <div class="sources-label">📚 Sources (${data.sources.length})</div>${cards}</div>`;
  }
  div.innerHTML=`<div class="avatar ai-av">🤖</div>
    <div class="bubble">
      <div class="bubble-text">${escHtml(data.answer||'No answer generated.')}</div>
      ${sourcesHtml}
    </div>`;
  area.appendChild(div);area.scrollTop=area.scrollHeight;
}

function toggleSource(header){
  const body=header.nextElementSibling;
  const chevron=header.querySelector('.chevron');
  chevron.classList.toggle('open',body.classList.toggle('open'));
}

async function refreshDocs(){
  const d=await fetch('/stats').then(r=>r.json());
  const list=document.getElementById('doc-list');
  const scope=document.getElementById('scope-select');
  const curVal=scope.value;
  if(!d.doc_details||!d.doc_details.length){
    list.innerHTML='<div class="empty-docs">No documents indexed</div>';
    scope.innerHTML='<option value="">All Documents</option>';return;
  }
  list.innerHTML=d.doc_details.map(doc=>`
    <div class="doc-item">
      <span class="doc-icon">📄</span>
      <div class="doc-info">
        <div class="doc-name" title="${escHtml(doc.filename)}">${escHtml(doc.filename)}</div>
        <div class="doc-meta">${doc.pages||'?'} pages · ${doc.chunks} chunks · ${doc.size_mb} MB</div>
      </div>
      <button class="doc-del" onclick="deleteDoc('${escHtml(doc.filename)}')" title="Remove">✕</button>
    </div>`).join('');
  scope.innerHTML='<option value="">All Documents</option>'+
    d.doc_details.map(doc=>`<option value="${escHtml(doc.filename)}"
      ${curVal===doc.filename?'selected':''}>${escHtml(doc.filename)}</option>`).join('');
}

async function deleteDoc(filename){
  if(!confirm(`Remove "${filename}" from the index?`))return;
  const d=await fetch('/delete_doc',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({filename})}).then(r=>r.json());
  showToast(d.success?`🗑 Removed: ${filename}`:`❌ ${d.error}`,d.success?'info':'error');
  refreshStats();refreshDocs();
}

async function refreshStats(){
  const d=await fetch('/stats').then(r=>r.json());
  document.getElementById('hdr-chunks').textContent=d.total_chunks||0;
  document.getElementById('hdr-docs').textContent=(d.documents||[]).length;
}

async function clearHistory(){
  await fetch('/clear_history',{method:'POST'});
  turnCount=0;document.getElementById('hdr-turns').textContent=0;
  document.getElementById('chat-area').innerHTML=
    '<div class="welcome" id="welcome" style="display:block"><div class="welcome-icon">🧹</div><h2>Chat Cleared</h2><p>History reset. Ask a new question.</p></div>';
  showToast('Chat history cleared','info');
}

async function resetAll(){
  if(!confirm('Reset everything? All documents and history will be cleared.'))return;
  await fetch('/reset',{method:'POST'});
  turnCount=0;
  document.getElementById('doc-list').innerHTML='<div class="empty-docs">No documents indexed</div>';
  document.getElementById('scope-select').innerHTML='<option value="">All Documents</option>';
  document.getElementById('chat-area').innerHTML=
    '<div class="welcome" id="welcome"><div class="welcome-icon">♻️</div><h2>Reset Complete</h2><p>Upload new documents to get started.</p></div>';
  refreshStats();showToast('System reset','info');
}

function showToast(msg,type='info'){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className=`toast ${type}`;t.style.display='block';
  setTimeout(()=>t.style.display='none',3500);
}

function escHtml(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/set_key", methods=["POST"])
def set_key():
    global _api_key
    data = request.get_json() or {}
    key  = data.get("key", "").strip()
    if not key.startswith("sk-"):
        return jsonify({"valid": False, "error": "Key must start with sk-"})
    try:
        payload = json.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": 5,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=10): pass
        _api_key = key
        return jsonify({"valid": True})
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return jsonify({"valid": False, "error": "Invalid API key"})
        _api_key = key   # rate limit or other — accept anyway
        return jsonify({"valid": True})
    except Exception:
        _api_key = key
        return jsonify({"valid": True})


@app.route("/key_status")
def key_status():
    return jsonify({"set": bool(_api_key)})


@app.route("/auto_index", methods=["POST"])
def auto_index():
    pdf = Path(__file__).parent / "data" / "documents" / "DIPAnn.pdf"
    if pdf.exists() and not engine.documents:
        result = engine.index_file(str(pdf))
        return jsonify(result)
    return jsonify({"success": False, "reason": "already indexed or file missing"})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"})
    f    = request.files["file"]
    name = f.filename or "upload"
    ext  = Path(name).suffix.lower()
    if ext not in {".pdf", ".txt", ".docx"}:
        return jsonify({"success": False,
                        "error": f"Unsupported type '{ext}'. Use PDF, DOCX, or TXT.",
                        "filename": name})
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        result = engine.index_file(tmp.name)
        if result.get("success"):
            temp_name = result["filename"]
            for i, src in enumerate(engine.chunk_sources):
                if src == temp_name: engine.chunk_sources[i] = name
            for doc in engine.documents:
                if doc["filename"] == temp_name: doc["filename"] = name
        os.unlink(tmp.name)
    result["filename"] = name
    return jsonify(result)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Retrieve relevant chunks + generate answer — all server-side.
    Works for any document domain (Soviet history, wildlife, medicine, anything).
    """
    global _api_key
    data     = request.get_json() or {}
    question = data.get("question", "").strip()
    scope    = data.get("source_filter") or None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not engine.chunks:
        return jsonify({"answer": "No documents indexed. Please upload a document first.",
                        "sources": []})

    if not _api_key:
        return jsonify({"error": "No API key set. Please enter your Anthropic API key in the sidebar."})

    # Retrieve chunks
    if _is_meta_query(question):
        all_doc_chunks = ([c for c, s in zip(engine.chunks, engine.chunk_sources) if s == scope]
                          if scope else list(engine.chunks))
        raw_info   = _extract_author_lines(all_doc_chunks)
        search_res = engine.search(question, top_k=6, source_filter=scope)
        meta_chunk = f"Extracted author/affiliation information:\n{raw_info}" if raw_info else ""
        chunks     = ([meta_chunk] if meta_chunk else []) + [r["text"] for r in search_res]
        results    = search_res
    else:
        results = engine.search(question, top_k=8, source_filter=scope)
        chunks  = [r["text"] for r in results]

    if not results:
        return jsonify({"answer": "No relevant information found in the selected documents.",
                        "sources": []})

    # Generate answer via Anthropic API (server-side)
    try:
        answer = _call_claude(question, chunks, _api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 401:
            _api_key = ""
            return jsonify({"error": "API key is invalid. Please re-enter it in the sidebar."})
        if e.code == 400 and "credit" in body.lower():
            return jsonify({"error": "Your Anthropic account has no credits. Please add credits at console.anthropic.com/settings/billing"})
        return jsonify({"error": f"Anthropic API error {e.code}: {body[:200]}"})
    except Exception as e:
        return jsonify({"error": f"Could not reach Anthropic API: {str(e)}"})

    # Build sources list
    sources_seen, sources = set(), []
    for r in results:
        key = f"{r['source']}_p{r['page']}"
        if key not in sources_seen:
            sources_seen.add(key)
            sources.append({"source": r["source"], "page": r["page"],
                            "score_pct": r["score_pct"], "excerpt": r["text"][:400]})

    return jsonify({"answer": answer, "sources": sources})


@app.route("/stats")
def stats():
    return jsonify(engine.get_stats())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    engine.clear_history()
    return jsonify({"success": True})


@app.route("/delete_doc", methods=["POST"])
def delete_doc():
    data     = request.get_json() or {}
    filename = data.get("filename", "")
    ok       = engine.remove_document(filename)
    return jsonify({"success": ok, "error": "" if ok else "Document not found"})


@app.route("/reset", methods=["POST"])
def reset():
    engine.reset()
    return jsonify({"success": True})


if __name__ == "__main__":
    if _api_key:
        print("  ✅ API key loaded from environment")
    else:
        print("  ⚠  Enter your API key in the sidebar after opening the app.")
    print("\n" + "═"*55)
    print("  🔍 RAG Document Assistant")
    print("  Upload any document, ask any question.")
    print("  Open → http://localhost:5000")
    print("═"*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)