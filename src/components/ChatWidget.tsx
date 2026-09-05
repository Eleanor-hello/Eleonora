import React, { useEffect, useRef } from 'react';
import { Message } from '../types';
import { Volume2 } from 'lucide-react';

interface ChatWidgetProps {
  messages: Message[];
  isThinking: boolean;
  onSpeak?: (text: string) => void;
}

export const ChatWidget: React.FC<ChatWidgetProps> = ({ messages, isThinking, onSpeak }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  return (
    <div
      id="chat-widget"
      ref={scrollRef}
      className="flex-1 overflow-y-auto p-4 space-y-3 bg-[#131324] rounded-[14px] border border-[#1E1E4A] transition-all"
    >
      {messages.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center text-center p-6 text-[#8B8FA3]">
          <div className="w-16 h-16 rounded-full bg-[#1A1A3E] border border-[#8B5CF6]/40 flex items-center justify-center text-2xl mb-3 shadow-lg shadow-[#8B5CF6]/10">
            💜
          </div>
          <h3 className="text-base font-semibold text-[#E2E8F0] mb-1">Элеонора v3</h3>
          <p className="text-xs max-w-sm">
            Голосовой ИИ-компаньон с адаптивной памятью, контролем ударений и естественной речью.
          </p>
        </div>
      ) : (
        messages.map((msg) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={msg.id}
              className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] md:max-w-[65%] rounded-[14px] p-3 text-[13.5px] leading-relaxed select-text shadow-sm relative group ${
                  isUser
                    ? 'bg-[#8B5CF6] text-white'
                    : 'bg-[#1A1A3E] text-[#E2E8F0] border border-[#1E1E4A]'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>
                <div className="flex items-center justify-between mt-1 gap-2 text-[10px] opacity-70">
                  <span>
                    {new Date(msg.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span>
                  {!isUser && onSpeak && (
                    <button
                      id={`speak-btn-${msg.id}`}
                      type="button"
                      onClick={() => onSpeak(msg.content)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 hover:text-white"
                      title="Озвучить"
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })
      )}

      {isThinking && (
        <div className="flex justify-start">
          <div className="bg-[#1A1A3E] text-[#06B6D4] border border-[#1E1E4A] rounded-[14px] px-4 py-2.5 text-xs flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#06B6D4] animate-ping" />
            <span>Элеонора думает...</span>
          </div>
        </div>
      )}
    </div>
  );
};
