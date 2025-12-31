import React, { useState, useRef, useEffect } from 'react';
import { useRAGQuery } from '../../hooks/useRAGQuery';
import styles from './styles.module.css';

export default function Chatbot(): React.JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [context, setContext] = useState('');
  const [useContextOnly, setUseContextOnly] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { loading, answer, sources, error, query, clear } = useRAGQuery();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [answer, loading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    await query(question, context, useContextOnly, null);
  };

  const handleClose = () => {
    setIsOpen(false);
    clear();
    setQuestion('');
    setContext('');
    setUseContextOnly(false);
  };

  return (
    <>
      {/* Floating Button */}
      <button
        className={styles.floatingButton}
        onClick={() => setIsOpen(true)}
        aria-label="Open chatbot"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </button>

      {/* Chat Modal */}
      {isOpen && (
        <div className={styles.overlay} onClick={handleClose}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            {/* Header */}
            <div className={styles.header}>
              <div className={styles.headerIcon}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <div className={styles.headerText}>
                <h3>Physical AI Assistant</h3>
                <p>Ask about humanoid robots, ROS 2, simulation, and VLA systems</p>
              </div>
              <button className={styles.closeButton} onClick={handleClose} aria-label="Close">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Chat Area */}
            <div className={styles.chatArea}>
              {!answer && !loading && !error && (
                <div className={styles.welcome}>
                  <div className={styles.welcomeIcon}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 16v-4M12 8h.01" />
                    </svg>
                  </div>
                  <h4>How can I help you learn?</h4>
                  <p>Ask questions about Physical AI, humanoid robotics, ROS 2, simulation, or VLA systems.</p>
                  <div className={styles.suggestions}>
                    <button onClick={() => setQuestion('What is Physical AI?')}>
                      What is Physical AI?
                    </button>
                    <button onClick={() => setQuestion('How does ROS 2 work?')}>
                      How does ROS 2 work?
                    </button>
                    <button onClick={() => setQuestion('What are VLA systems?')}>
                      What are VLA systems?
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className={styles.error}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 8v4M12 16h.01" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              {answer && (
                <div className={styles.message}>
                  <div className={styles.answer}>
                    <div
                      className={styles.answerContent}
                      dangerouslySetInnerHTML={{
                        __html: answer
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/### (.*?)$/gm, '<h4>$1</h4>')
                          .replace(/## (.*?)$/gm, '<h3>$1</h3>')
                          .replace(/^\d+\.\s+\*\*(.*?)\*\*:/gm, '<strong>$1</strong>:')
                          .replace(/^- \*\* (.*?):/gm, '<li><strong>$1</strong>:')
                          .replace(/^- (.*?)$/gm, '<li>$1</li>')
                          .replace(/<li>/g, '<ul><li>').replace(/<\/li>/g, '</li></ul>')
                          .replace(/<ul><\/ul>/g, '')
                          .replace(/\n\n/g, '<br/><br/>')
                      }}
                    />
                  </div>
                  {sources.length > 0 && (
                    <div className={styles.sources}>
                      <div className={styles.sourcesHeader}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                          <line x1="16" y1="13" x2="8" y2="13" />
                          <line x1="16" y1="17" x2="8" y2="17" />
                        </svg>
                        <span>Sources</span>
                      </div>
                      {sources.slice(0, 3).map((source, idx) => (
                        <div key={idx} className={styles.sourceItem}>
                          <span className={styles.sourceChapter}>Ch. {source.chapter_id}</span>
                          <span className={styles.sourceTitle}>{source.section_title}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {loading && (
                <div className={styles.loading}>
                  <div className={styles.spinner} />
                  <span>Thinking...</span>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Context Input (Optional) */}
            {useContextOnly && (
              <div className={styles.contextInput}>
                <label>Selected text context:</label>
                <textarea
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  placeholder="Paste text from the textbook..."
                  rows={3}
                />
              </div>
            )}

            {/* Input Form */}
            <form onSubmit={handleSubmit} className={styles.inputForm}>
              <div className={styles.options}>
                <label className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={useContextOnly}
                    onChange={(e) => setUseContextOnly(e.target.checked)}
                  />
                  <span>Use selected text only</span>
                </label>
              </div>
              <div className={styles.inputRow}>
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question..."
                  disabled={loading}
                />
                <button type="submit" disabled={loading || !question.trim()}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
