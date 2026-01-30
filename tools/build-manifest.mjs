import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const CODE_DIR = path.join(ROOT, "code");
const OUT_DIR = path.join(ROOT, "public");
const FILES_DIR = path.join(OUT_DIR, "files");

function walk(dir){
  const out = [];
  const items = fs.readdirSync(dir, { withFileTypes:true });
  for (const it of items){
    // 可按需忽略：比如 .venv、__pycache__、node_modules 等
    if ([".venv","__pycache__",".git","node_modules",".DS_Store"].includes(it.name)) continue;

    const abs = path.join(dir, it.name);
    if (it.isDirectory()){
      out.push(...walk(abs));
    }else{
      out.push(abs);
    }
  }
  return out;
}

function ensureDir(p){
  fs.mkdirSync(p, { recursive:true });
}

function copyFilePreserveDirs(srcAbs){
  const rel = path.relative(CODE_DIR, srcAbs).replaceAll("\\","/");
  const dstAbs = path.join(FILES_DIR, rel);
  ensureDir(path.dirname(dstAbs));
  fs.copyFileSync(srcAbs, dstAbs);
  return rel;
}

function fileSize(abs){
  try{ return fs.statSync(abs).size; }catch{ return 0; }
}

function main(){
  if (!fs.existsSync(CODE_DIR)){
    console.error("找不到 code/ 文件夹。请把要展示的代码放到 code/ 目录下。");
    process.exit(1);
  }

  // 清空 public/files
  if (fs.existsSync(FILES_DIR)){
    fs.rmSync(FILES_DIR, { recursive:true, force:true });
  }
  ensureDir(FILES_DIR);

  const files = walk(CODE_DIR);
  const nodes = [];

  // 先把目录节点也写入（可选，但方便树结构）
  const dirSet = new Set();
  for (const f of files){
    const rel = path.relative(CODE_DIR, f).replaceAll("\\","/");
    const parts = rel.split("/");
    for (let i=0;i<parts.length-1;i++){
      const d = parts.slice(0,i+1).join("/");
      dirSet.add(d);
    }
  }
  for (const d of Array.from(dirSet).sort()){
    nodes.push({ path:d, type:"dir", size:0 });
  }

  // 文件节点 + 复制
  for (const abs of files){
    const rel = copyFilePreserveDirs(abs);
    nodes.push({ path: rel, type:"file", size: fileSize(abs) });
  }

  const manifest = {
    generatedAt: new Date().toISOString(),
    nodes
  };

  ensureDir(OUT_DIR);
  fs.writeFileSync(path.join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2), "utf-8");
  console.log(`OK: ${nodes.length} nodes, ${files.length} files copied to public/files/`);
}

main();
