import React from 'react';
import { ChatSession } from '../types';
import { X, MessageSquare, Trash2, Plus } from 'lucide-react';

interface SessionsModalProps {
  isOpen: boolean;
  sessions: ChatSession[];
  currentSessionId: number | null;
  onSelectSession: (id: number) => void;
  onDeleteSession: (id: number) => void;
  onNewSession: () => void;
  onClose: () => void;
}

export const SessionsModal: React.FC<SessionsModalProps> = ({
  isOpen,
  sessions,
  currentSessionId,
  onSelectSession,
  onDeleteSession,
  onNewSession,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div
        id="sessions-modal"
        className="bg-[#131324] border border-[#8B5CF6]/50 rounded-2xl w-full max-w-md max-h-[80vh] flex flex-col shadow-2xl overflow-hidden"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#1E1E4A] bg-[#1A1A3E]">
          <div className="flex items-center gap-2">
            <span className="text-lg">📜</span>
            <h3 className="font-semibold text-sm text-[#E2E8F0]">История диалогов</h3>
          </div>
          <button
            id="close-sessions-modal-btn"
            type="button"
            onClick={onClose}
            className="p-1 rounded-full text-[#8B8FA3] hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-3 border-b border-[#1E1E4A] flex justify-between items-center">
          <span className="text-xs text-[#8B8FA3]">Всего сессий: {sessions.length}</span>
          <button
            id="modal-new-chat-btn"
            type="button"
            onClick={() => {
              onNewSession();
              onClose();
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#8B5CF6] hover:bg-[#7C3AED] text-white rounded-lg text-xs font-medium transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Новый диалог</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {sessions.length === 0 ? (
            <div className="text-center py-8 text-xs text-[#8B8FA3]">
              Нет сохранённых диалогов
            </div>
          ) : (
            sessions.map((s) => {
              const isCurrent = s.id === currentSessionId;
              const dateStr = s.updatedAt
                ? new Date(s.updatedAt).toLocaleString([], {
                    dateStyle: 'short',
                    timeStyle: 'short',
                  })
                : '';

              return (
                <div
                  key={s.id}
                  className={`flex items-center justify-between p-2.5 rounded-xl border transition-all cursor-pointer ${
                    isCurrent
                      ? 'bg-[#1A1A3E] border-[#8B5CF6] text-white shadow-md shadow-[#8B5CF6]/10'
                      : 'bg-[#0B0B16] border-[#1E1E4A] text-[#E2E8F0] hover:border-[#8B5CF6]/40'
                  }`}
                  onClick={() => {
                    onSelectSession(s.id);
                    onClose();
                  }}
                >
                  <div className="flex items-center gap-3 overflow-hidden">
                    <MessageSquare className={`w-4 h-4 shrink-0 ${isCurrent ? 'text-[#8B5CF6]' : 'text-[#8B8FA3]'}`} />
                    <div className="overflow-hidden">
                      <div className="text-xs font-medium truncate">{s.title}</div>
                      <div className="text-[10px] text-[#8B8FA3]">{dateStr}</div>
                    </div>
                  </div>

                  <button
                    id={`delete-session-${s.id}`}
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(s.id);
                    }}
                    className="p-1.5 rounded-lg text-[#8B8FA3] hover:text-red-400 hover:bg-red-400/10 transition-colors shrink-0 ml-2"
                    title="Удалить диалог"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
