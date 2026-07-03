import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      // Use the full Vue build (runtime + compiler) so in-DOM HTML templates compile
      'vue': 'vue/dist/vue.esm-bundler.js'
    }
  },
  server: {
    port: 5174,
    strictPort: false, // if 5174 is taken, try next port
    allowedHosts: ['emptier-bounding-bullring.ngrok-free.dev', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  }
})
