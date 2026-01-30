import { defineConfig } from "vite";

export default defineConfig({
  // GitHub Pages 如果是 https://用户名.github.io/仓库名/
  // 那 base 需要设置为 "/仓库名/"。先写成 "./" 让它相对路径，也能用。
  base: "./",
  root: "public",
  build: {
    outDir: "../dist",
    emptyOutDir: true
  }
});
