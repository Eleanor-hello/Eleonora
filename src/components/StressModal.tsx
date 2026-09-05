import React, { useState } from 'react';
import { StressOverride } from '../types';
import { X, Plus, Trash2, BookOpen } from 'lucide-react';

interface StressModalProps {
  isOpen: boolean;
  overrides: StressOverride[];
  onAddOverride: (marked: string) => Promise<void>;
  onDeleteOverride: (bare: string) => Promise<void>;
  onClose: () => void;
}

export const StressModal: React.FC<StressModalProps> = ({
  isOpen,
  overrides,
  onAddOverride,
  onDeleteOverride,
  onClose,
}) => {
  const [wordInput, setWordInput] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!wordInput.includes('+')) {
      setError("Поставьте '+' перед ударной гласной (например: молок+о)");
      return;
    }
    try {
      setIsSubmitting(true);
      setError('');
      await onAddOverride(wordInput.trim());
      setWordInput('');
    } catch (err: any) {
      setError(err.message || 'Ошибка сохранения');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div
        id="stress-modal"
        className="bg-[#131324] border border-[#8B5CF6]/50 rounded-2xl w-full max-w-md max-h-[80vh] flex flex-col shadow-2xl overflow-hidden"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#1E1E4A] bg-[#1A1A3E]">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-[#8B5CF6]" />
            <h3 className="font-semibold text-sm text-[#E2E8F0]">База ударений (Stress Check)</h3>
          </div>
          <button
            id="close-stress-modal-btn"
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
                id="stress-word-input"
                type="text"
                placeholder="Слово со знаком + (напр. зв+онит)"
                value={wordInput}
                onChange={(e) => {
                  setWordInput(e.target.value);
                  setError('');
                }}
                className="flex-1 bg-[#131324] border border-[#1E1E4A] rounded-lg px-3 py-1.5 text-xs text-[#E2E8F0] focus:border-[#8B5CF6] outline-none"
              />
              <button
                id="add-stress-btn"
                type="submit"
                disabled={isSubmitting || !wordInput}
                className="px-3 py-1.5 bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:bg-[#1E1E4A] text-white rounded-lg text-xs font-medium flex items-center gap-1 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Добавить</span>
              </button>
            </div>
            {error && <div className="text-[11px] text-red-400">{error}</div>}
            <div className="text-[10px] text-[#8B8FA3]">
              Элеонора автоматически запоминает ударения, когда в сообщении написано «не X, а Y» или «слово+».
            </div>
          </form>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
          {overrides.length === 0 ? (
            <div className="text-center py-6 text-xs text-[#8B8FA3]">
              Нет сохранённых ударений
            </div>
          ) : (
            overrides.map((item) => (
              <div
                key={item.bare}
                className="flex items-center justify-between p-2 rounded-lg bg-[#0B0B16] border border-[#1E1E4A] text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[#8B8FA3]">{item.bare}</span>
                  <span className="text-[#06B6D4]">→</span>
                  <span className="font-semibold text-[#8B5CF6]">{item.marked}</span>
                </div>
                <button
                  id={`delete-stress-${item.bare}`}
                  type="button"
                  onClick={() => onDeleteOverride(item.bare)}
                  className="text-[#8B8FA3] hover:text-red-400 p-1 transition-colors"
                  title="Удалить"
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
