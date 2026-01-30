let manifest = null;
let currentPath = null;
let fileCache = new Map();

const elTree = document.getElementById("tree");
const elCode = document.getElementById("code");
const elFilepath = document.getElementById("filepath");
const elMeta = document.getElementById("meta");
const elSearch = document.getElementById("search");
const elCopyBtn = document.getElementById("copyBtn");
const elRawLink = document.getElementById("rawLink");

function iconFor(type){
  return type === "dir" ? "📁" : "📄";
}

function ext(path){
  const i = path.lastIndexOf(".");
  return i >= 0 ? path.slice(i+1).toLowerCase() : "";
}

function prismLangByExt(e){
  // Prism 默认不含所有语言高亮，这里先做“可用就用”的映射
  const map = {
    js:"javascript", mjs:"javascript", cjs:"javascript",
    ts:"typescript",
    py:"python",
    java:"java",
    c:"c", cpp:"cpp", h:"c", hpp:"cpp",
    json:"json",
    yml:"yaml", yaml:"yaml",
    md:"markdown",
    html:"markup", css:"css",
    sh:"bash", bash:"bash"
  };
  return map[e] || "none";
}

function buildTree(nodes){
  // nodes: [{path, type, size}]
  const root = { name:"", type:"dir", children:new Map(), path:"" };

  for (const n of nodes){
    const parts = n.path.split("/").filter(Boolean);
    let cur = root;
    for (let i=0;i<parts.length;i++){
      const name = parts[i];
      const isLast = i === parts.length - 1;
      const type = isLast ? n.type : "dir";
      if (!cur.children.has(name)){
        cur.children.set(name, { name, type, children:new Map(), path: parts.slice(0,i+1).join("/") });
      }
      cur = cur.children.get(name);
      if (isLast){
        cur.size = n.size ?? 0;
        cur.type = n.type;
      }
    }
  }
  return root;
}

function renderNode(node, container, depth=0, filter=""){
  const children = Array.from(node.children.values())
    .sort((a,b)=>{
      if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

  for (const child of children){
    const full = child.path;
    const match = !filter || full.toLowerCase().includes(filter);

    // 如果是文件夹，只有当它自己或其子项匹配才显示
    let showDir = match;
    if (child.type === "dir" && filter){
      showDir = hasMatchInSubtree(child, filter);
    }
    if (!showDir && child.type === "dir") continue;
    if (!match && child.type === "file") continue;

    const row = document.createElement("div");
    row.className = "node" + (currentPath === full ? " active" : "");
    row.style.marginLeft = (depth * 2) + "px";
    row.dataset.path = full;
    row.dataset.type = child.type;
    row.innerHTML = `
      <span class="ico">${iconFor(child.type)}</span>
      <span class="name" title="${full}">${child.name}</span>
      ${child.type === "file" ? `<span class="badge">${formatBytes(child.size || 0)}</span>` : `<span class="badge">dir</span>`}
    `;

    row.addEventListener("click", async ()=>{
      if (child.type === "dir"){
        // 目录点击只展开视觉上不需要额外逻辑（静态渲染）
        return;
      }
      await openFile(full, child.size || 0);
    });

    container.appendChild(row);

    if (child.type === "dir"){
      const sub = document.createElement("div");
      sub.className = "indent";
      container.appendChild(sub);
      renderNode(child, sub, depth+1, filter);
    }
  }
}

function hasMatchInSubtree(node, filter){
  const q = filter.toLowerCase();
  for (const child of node.children.values()){
    const full = child.path.toLowerCase();
    if (full.includes(q)) return true;
    if (child.type === "dir" && hasMatchInSubtree(child, filter)) return true;
  }
  return false;
}

function rerenderTree(){
  elTree.innerHTML = "";
  const root = buildTree(manifest.nodes);
  renderNode(root, elTree, 0, elSearch.value.trim().toLowerCase());
}

function formatBytes(n){
  if (!n) return "0 B";
  const units = ["B","KB","MB","GB"];
  let i=0, x=n;
  while (x>=1024 && i<units.length-1){ x/=1024; i++; }
  return `${x.toFixed(x>=10||i===0?0:1)} ${units[i]}`;
}

async function openFile(path, size){
  currentPath = path;
  const rawUrl = `./files/${encodeURIComponent(path).replaceAll("%2F","/")}`;

  elFilepath.textContent = path;
  elMeta.textContent = `Size: ${formatBytes(size)}`;

  elCopyBtn.disabled = false;
  elRawLink.setAttribute("aria-disabled","false");
  elRawLink.href = rawUrl;

  // URL hash 方便分享定位
  location.hash = `#file=${encodeURIComponent(path)}`;

  // 读取文件内容（静态网站 fetch）
  let text = fileCache.get(path);
  if (!text){
    const res = await fetch(rawUrl);
    if (!res.ok){
      elCode.textContent = `无法读取文件：${res.status} ${res.statusText}`;
      return;
    }
    text = await res.text();
    fileCache.set(path, text);
  }

  // 设置 Prism 语言
  const e = ext(path);
  const lang = prismLangByExt(e);
  elCode.className = `language-${lang}`;
  elCode.textContent = text;

  // 高亮（CDN 版 Prism 只有基础语言可用；没高亮也不影响浏览）
  if (window.Prism && Prism.highlightElement){
    Prism.highlightElement(elCode);
  }

  // 更新树高亮
  rerenderTree();
}

elCopyBtn.addEventListener("click", async ()=>{
  if (!currentPath) return;
  const text = fileCache.get(currentPath) || elCode.textContent || "";
  await navigator.clipboard.writeText(text);
  elCopyBtn.textContent = "已复制 ✓";
  setTimeout(()=> elCopyBtn.textContent = "复制当前文件", 900);
});

elSearch.addEventListener("input", ()=>{
  rerenderTree();
});

async function init(){
  const res = await fetch("./manifest.json");
  manifest = await res.json();
  rerenderTree();

  // 如果打开网址时带了 #file=xxx，则自动打开该文件
  const m = location.hash.match(/file=([^&]+)/);
  if (m){
    const p = decodeURIComponent(m[1]);
    const node = manifest.nodes.find(x => x.path === p && x.type === "file");
    if (node) await openFile(p, node.size || 0);
  }
}
init();
