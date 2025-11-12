import { useMutation } from '@tanstack/react-query';
import clsx from 'clsx';
import { useMemo, useState } from 'react';

import { fetchComparison } from '../api/client';
import type { ConversationTurn } from '../types';

const defaultPrompt =
  'USER: hola necesito ayuda con un problema familiar muy complicado BOT: claro dime más USER: llevo semanas intentando hablar pero no me entienden y me siento aislado BOT: entiendo debe ser duro USER: cada conversación termina en discusiones y estoy agotado';

export function ChatView() {
  const [message, setMessage] = useState('');
  const [maxNew, setMaxNew] = useState(60);
  const [temperature, setTemperature] = useState(0.8);
  const [history, setHistory] = useState<ConversationTurn[]>([]);

  const mutation = useMutation({
    mutationFn: fetchComparison,
    onSuccess: (data) => {
      const turn: ConversationTurn = {
        id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
        prompt: data.input,
        emoResponse: data.emo_output,
        alignment: data.emo_alignment,
        chatgptResponse: data.chatgpt_output,
        chatgptModel: data.chatgpt_model,
        warnings: data.warnings,
        createdAt: new Date().toISOString()
      };
      setHistory((prev) => [turn, ...prev]);
    }
  });

  const handleSubmit = (evt: React.FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    const trimmed = message.trim();
    mutation.mutate({
      message: trimmed,
      max_new: maxNew,
      temp: temperature
    });
    if (!trimmed) {
      setMessage('');
    }
  };

  const lastWarnings = useMemo(() => history[0]?.warnings ?? [], [history]);

  return (
    <div className="flex w-full max-w-6xl flex-col gap-6 px-6 py-10">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold text-white">Comparador emocional & ChatGPT</h1>
        <p className="text-sm text-slate-300">
          Envía un mensaje (o historial completo) y compara la respuesta del modelo emocional con la de ChatGPT.
        </p>
      </header>

      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-4 rounded-xl bg-slate-800/60 p-5 shadow-lg shadow-indigo-900/20"
      >
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium uppercase tracking-wide text-slate-300">Mensaje / historial</span>
          <textarea
            value={message}
            onChange={(evt) => setMessage(evt.target.value)}
            placeholder={defaultPrompt}
            rows={4}
            className="rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary focus:ring-2 focus:ring-primary/60"
          />
        </label>

        <div className="grid gap-4 sm:grid-cols-3">
          <label className="flex flex-col gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-400">Tokens máximos</span>
            <input
              type="number"
              min={10}
              max={200}
              value={maxNew}
              onChange={(evt) => setMaxNew(Number(evt.target.value))}
              className="rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary focus:ring-2 focus:ring-primary/60"
            />
          </label>

          <label className="flex flex-col gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-400">Temperatura</span>
            <input
              type="number"
              step="0.05"
              min={0.1}
              max={1.5}
              value={temperature}
              onChange={(evt) => setTemperature(Number(evt.target.value))}
              className="rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary focus:ring-2 focus:ring-primary/60"
            />
          </label>

          <div className="flex items-end">
            <button
              type="submit"
              disabled={mutation.isLoading}
              className={clsx(
                'w-full rounded-lg bg-gradient-to-r from-primary to-secondary px-4 py-2 text-sm font-semibold shadow-lg shadow-indigo-900/30 transition-all',
                mutation.isLoading && 'opacity-60'
              )}
            >
              {mutation.isLoading ? 'Generando...' : 'Comparar respuestas'}
            </button>
          </div>
        </div>

        {mutation.isError && (
          <p className="rounded-md border border-red-500 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            Ocurrió un error al solicitar la comparación. Revisa los logs del backend.
          </p>
        )}
      </form>

      {lastWarnings.length > 0 && (
        <div className="rounded-lg border border-amber-500/60 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {lastWarnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      )}

      <section className="flex flex-col gap-4 pb-12">
        {history.length === 0 && (
          <p className="text-center text-sm text-slate-400">
            Aún no hay conversaciones. Envía tu primer mensaje para ver la comparación.
          </p>
        )}

        {history.map((turn) => (
          <article
            key={turn.id}
            className="rounded-xl border border-slate-700/60 bg-slate-800/70 p-5 shadow-lg shadow-slate-900/30"
          >
            <header className="mb-4 flex flex-col gap-1 text-sm text-slate-300">
              <span className="font-semibold text-slate-100">Usuario</span>
              <p className="whitespace-pre-line text-slate-200">{turn.prompt}</p>
              <span className="text-xs text-slate-500">
                {new Date(turn.createdAt).toLocaleString()} · Alineación emocional:{' '}
                {(turn.alignment * 100).toFixed(1)}%
              </span>
            </header>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-primary/40 bg-primary/10 p-4">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">Modelo emocional</h3>
                <p className="mt-2 text-sm text-slate-100 whitespace-pre-line">{turn.emoResponse}</p>
              </div>
              <div className="rounded-lg border border-secondary/40 bg-secondary/10 p-4">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-secondary">
                  ChatGPT {turn.chatgptModel ? `(${turn.chatgptModel})` : ''}
                </h3>
                <p className="mt-2 text-sm text-slate-100 whitespace-pre-line">
                  {turn.chatgptResponse ?? 'Sin respuesta (revisa las advertencias).'}
                </p>
              </div>
            </div>
            {turn.warnings.length > 0 && (
              <ul className="mt-3 list-disc space-y-1 pl-6 text-xs text-amber-200">
                {turn.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
          </article>
        ))}
      </section>
    </div>
  );
}
