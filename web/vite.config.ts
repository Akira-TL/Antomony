import {defineConfig} from 'vite';

export default defineConfig({
  build: {rollupOptions: {input: {colony: 'index.html', training: 'training.html', recurrent: 'recurrent.html', roundtrip: 'roundtrip.html', acceptance: 'acceptance.html', interactive:'interactive.html'}}},
});
