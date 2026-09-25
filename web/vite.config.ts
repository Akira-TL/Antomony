import {defineConfig} from 'vite';

export default defineConfig({
  build: {rollupOptions: {input: {colony: 'index.html', training: 'training.html'}}},
});
