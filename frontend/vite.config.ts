import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',                      // allow external connections
    port: 3000,                            // run on port 3000 — different from PitchIQ
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8001',  // port 8001 locally
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})


