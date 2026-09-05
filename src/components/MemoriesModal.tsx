import React, { useState } from 'react';
import { PersonalFact } from '../types';
import { X, Plus, Trash2, Brain } from 'lucide-react';

interface MemoriesModalProps {
  isOpen: boolean;
  memories: PersonalFact[];
  onAddMemory: (fact: string, category?: string) => Promise<void>;
  onDeleteMemory: (id: string) => Promise<void>;
  onClose: () => void;
}

export const MemoriesModal: React.FC<MemoriesModalProps> = ({
  isOpen,
  memories,
  onAddMemory,
  onDeleteMemory,
  onClose,
}) => {
  const [factInput, setFactInput] = useState('');
  const [category, setCategory] = useState('user');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!factInput.trim()) return;
    try {
      setIsSubmitting(true);
      await onAddMemory(factInput.trim(), category);
      setFactInput('');
    } catch (err) {
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div
        id="memories-modal"
        className="bg-[#131324] border border-[#8B5CF6]/50 rounded-2xl w-full max-w-md max-h-[80vh] flex flex-col shadow-2xl overflow-hidden"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#1E1E4A] bg-[#1A1A3E]">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-[#06B6D4]" />
            <h3 className="font-semibold text-sm text-[#E2E8F0]">Память и личные факты (Memory)</h3>
          </div>
          <button
            id="close-memories-modal-btn"
            type="button"
            onClick={onClose}
            className="p-1 rounded-full text-[#8B8FA3] hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-3 border-b border-[#1E1E4A] bg-[#0B0B16]">
          <form onSubmit={handleSubmit} className="space-y-2">
            <div className="flex gap-2">
              <input
                id="memory-fact-input"
                type="text"
                placeholder="Новый факт (напр. У меня есть кот Жужа)"
                value={factInput}
                onChange={(e) => setFactInput(e.target.value)}
                className="flex-1 bg-[#131324] border border-[#1E1E4A] rounded-lg px-3 py-1.5 text-xs text-[#E2E8F0] focus:border-[#8B5CF6] outline-none"
              />
              <button
                id="add-memory-btn"
                type="submit"
                disabled={isSubmitting || !factInput.trim()}
                className="px-3 py-1.5 bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:bg-[#1E1E4A] text-white rounded-lg text-xs font-medium flex items-center gap-1 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Добавить</span>
              </button>
            </div>
            <div className="text-[10px] text-[#8B8FA3]">
              Элеонора автоматически консолидирует факты из сообщений («меня зовут...», «я работаю...»).
            </div>
          </form>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {memories.length === 0 ? (
            <div className="text-center py-6 text-xs text-[#8B8FA3]">
              Память пуста. Расскажите Элеоноре что-нибудь о себе!
            </div>
          ) : (
            memories.map((m) => (
              <div
                key={m.id}
                className="flex items-start justify-between p-2.5 rounded-xl bg-[#0B0B16] border border-[#1E1E4A] text-xs gap-2"
              >
                <div className="flex-1">
                  <div className="text-[#E2E8F0]">{m.fact}</div>
                  {m.category && (
                    <span className="inline-block mt-1 px-1.5 py-0.5 rounded text-[9px] bg-[#1A1A3E] text-[#06B6D4]">
                      {m.category}
                    </span>
                  )}
                </div>
                <button
                  id={`delete-memory-${m.id}`}
                  type="button"
                  onClick={() => onDeleteMemory(m.id)}
                  className="text-[#8B8FA3] hover:text-red-400 p-1 transition-colors shrink-0"
                  title="Удалить факт"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
