import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx,jsx,js}'],
  theme: {
    extend: {
      colors: {
        primary: '#4F46E5',
        secondary: '#DB2777'
      }
    }
  },
  plugins: []
} satisfies Config;
