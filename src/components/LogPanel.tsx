import React, { useEffect, useRef } from 'react';
import { LogEntry } from '../types';
import { X, Trash2, ShieldAlert } from 'lucide-react';

interface LogPanelProps {
  logs: LogEntry[];
  isOpen: boolean;
  onClose: () => void;
  onClear: () => void;
}

export const LogPanel: React.FC<LogPanelProps> = ({ logs, isOpen, onClose, onClear }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isOpen]);

  if (!isOpen) return null;

  return (
    <div
      id="log-panel"
      className="h-56 bg-[#0B0B16] border border-[#1E1E4A] rounded-t-xl flex flex-col transition-all shadow-2xl relative z-20"
    >
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#131324] border-b border-[#1E1E4A] text-xs">
        <div className="flex items-center gap-2 text-[#E2E8F0] font-medium">
          <span>📋 Логи агентов и системы ({logs.length})</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            id="clear-logs-btn"
            type="button"
            onClick={onClear}
            className="text-[#8B8FA3] hover:text-[#E2E8F0] p-1 rounded transition-colors"
            title="Очистить логи"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          <button
            id="close-logs-btn"
            type="button"
            onClick={onClose}
            className="text-[#8B8FA3] hover:text-[#E2E8F0] p-1 rounded transition-colors"
            title="Закрыть панель логов"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-2.5 font-mono text-[11.5px] leading-relaxed space-y-1 select-text"
      >
        {logs.length === 0 ? (
          <div className="text-[#8B8FA3] italic p-2">Логов пока нет. Отправьте сообщение в чат.</div>
        ) : (
          logs.map((log) => {
            const isWarn = log.level === 'WARNING';
            const isErr = log.level === 'ERROR';
            const colorClass = isErr
              ? 'text-red-400'
              : isWarn
              ? 'text-amber-400'
              : 'text-[#8B8FA3]';

            return (
              <div key={log.id} className={`flex items-start gap-2 ${colorClass}`}>
                <span className="text-[#8B8FA3]/60 shrink-0">{log.timestamp}</span>
                <span className="font-semibold px-1 rounded bg-[#131324] text-[#06B6D4] text-[10px] shrink-0">
                  {log.component}
                </span>
                <span className="break-all">{log.message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
