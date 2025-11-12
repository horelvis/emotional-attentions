import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { ChatView } from './components/ChatView';

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex min-h-screen">
        <aside className="hidden w-80 flex-shrink-0 flex-col justify-between border-r border-white/10 bg-slate-900/60 px-6 py-8 md:flex">
          <div className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-white">Sesiones</h2>
              <p className="mt-1 text-xs text-slate-400">
                (En desarrollo) Gestiona múltiples conversaciones y ajustes rápidos desde aquí.
              </p>
            </div>
            <nav className="space-y-3 text-sm text-slate-300">
              <button className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-left font-medium text-slate-100 hover:border-primary/60 hover:bg-primary/10">
                Nueva conversación
              </button>
              <button className="w-full rounded-lg border border-white/5 bg-white/5 px-3 py-2 text-left text-slate-300 hover:border-secondary/60 hover:bg-secondary/10">
                Ajustes rápidos
              </button>
            </nav>
          </div>
          <footer className="text-xs text-slate-500">
            <p>EmoChat Compare v{__APP_VERSION__}</p>
            <p>Desarrollado con FastAPI + React.</p>
          </footer>
        </aside>
        <main className="flex flex-1 justify-center">
          <ChatView />
        </main>
      </div>
    </QueryClientProvider>
  );
}
