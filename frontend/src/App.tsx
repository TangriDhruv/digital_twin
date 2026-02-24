import { useState, useRef, useEffect, useCallback } from 'react'
const API_BASE = import.meta.env.VITE_API_URL ?? ''

// ── Types ──────────────────────────────────────────────────────────────────

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

interface SSEToken  { token: string }
interface SSEDone   { done: boolean }
interface SSEError  { error: string }
type SSEEvent = SSEToken | SSEDone | SSEError

// ── Constants ──────────────────────────────────────────────────────────────

const SUGGESTED_QUESTIONS: string[] = [
  "Tell me about yourself",
  "What's your experience with RAG pipelines?",
  "RAG vs fine-tuning — what do you prefer?",
  "What are you most excited about in AI?",
  "Walk me through your work at KPMG",
  "What does your ideal next role look like?",
]

const SKILLS: string[] = [
  'Python', 'SQL', 'FastAPI', 'React', 'LangGraph', 'FAISS', 'Docker'
]

// ── Sidebar ────────────────────────────────────────────────────────────────

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="profile-section">
        <div className="profile-badge">Digital Twin · Active</div>
        <h1 className="profile-name">Dhruv<br />Tangri</h1>
        <p className="profile-title">
          Data Scientist / AI Engineer<br />
          Carnegie Mellon University, MS '25
        </p>
      </div>

      <div className="divider" />

      <div className="meta-block">
        <p className="meta-label">Currently</p>
        <div className="meta-item">AI Engineer @ Analytics 4 Everyone</div>
        <div className="meta-item">MS BIDA @ CMU Heinz (GPA 3.96)</div>
      </div>

      <div className="meta-block">
        <p className="meta-label">Background</p>
        <div className="meta-item">4 years at KPMG — finance, audit, data</div>
        <div className="meta-item">Goldman Sachs client via KPMG</div>
        <div className="meta-item">BE Mechanical Eng. @ VIT</div>
      </div>

      <div className="meta-block">
        <p className="meta-label">Stack</p>
        <div className="tag-list">
          {SKILLS.map(s => <span key={s} className="tag">{s}</span>)}
        </div>
      </div>

      <div className="divider" />

      <div className="sidebar-footer">
        <div>dhruv1998tangri@gmail.com</div>
        <div>Pittsburgh, PA</div>
      </div>
    </aside>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────

function ThinkingIndicator() {
  return (
    <div className="thinking">
      <div className="thinking-dots">
        <span /><span /><span />
      </div>
      retrieving context...
    </div>
  )
}

function ChatMessage({ msg }: { msg: Message }) {
  return (
    <div className={`message ${msg.role}`}>
      <span className="message-label">
        {msg.role === 'user' ? 'You' : 'Dhruv'}
      </span>
      <div className="message-bubble">
        {msg.content}
        {msg.streaming && <span className="cursor" />}
      </div>
    </div>
  )
}

function EmptyState({ onSuggest }: { onSuggest: (q: string) => void }) {
  return (
    <div className="empty-state">
      <div className="empty-headline">
        <h2>Ask <em>Dhruv</em><br />anything.</h2>
        <p>// powered by his actual experiences and opinions</p>
      </div>
      <div className="suggestions-grid">
        {SUGGESTED_QUESTIONS.map(q => (
          <button key={q} className="suggestion-btn" onClick={() => onSuggest(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput]       = useState<string>('')
  const [loading, setLoading]   = useState<boolean>(false)
  const [error, setError]       = useState<string | null>(null)
  const bottomRef               = useRef<HTMLDivElement>(null)
  const textareaRef             = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea as user types
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 140) + 'px'
  }

  const sendMessage = useCallback(async (text?: string) => {
    const userText = (text ?? input).trim()
    if (!userText || loading) return

    setError(null)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    // Append user message
    const userMsg: Message = { id: Date.now(), role: 'user', content: userText }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    // Append empty assistant message to stream tokens into
    const assistantId = Date.now() + 1
    setMessages(prev => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', streaming: true }
    ])

    try {
      // History = all messages so far (excludes the streaming placeholder)
      const history = messages.map(m => ({ role: m.role, content: m.content }))
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText, history }),
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      if (!response.body)  throw new Error('No response body')

      // Read SSE stream token by token
      const reader  = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer    = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''   // keep any incomplete line in the buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6)) as SSEEvent

            if ('token' in data) {
              setMessages(prev => prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + data.token }
                  : m
              ))
            }

            if ('done' in data && data.done) {
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, streaming: false } : m
              ))
            }

            if ('error' in data) {
              setError(data.error)
              setMessages(prev => prev.filter(m => m.id !== assistantId))
            }

          } catch {
            // malformed JSON line — skip silently
          }
        }
      }

    } catch (err) {
      setError('Could not reach the server. Is the backend running on port 8000?')
      setMessages(prev => prev.filter(m => m.id !== assistantId))
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
    setError(null)
  }

  const isEmpty = messages.length === 0
  const isThinking = loading && messages.at(-1)?.content === ''

  return (
    <div className="app">
      <Sidebar />

      <div className="chat-area">
        {/* Header */}
        <div className="chat-header">
          <span className="chat-header-title">
            <span>twin</span>.chat / session
          </span>
          {!isEmpty && (
            <button className="clear-btn" onClick={clearChat}>
              Clear
            </button>
          )}
        </div>

        {/* Body */}
        {isEmpty ? (
          <EmptyState onSuggest={(q) => sendMessage(q)} />
        ) : (
          <div className="messages-container">
            {messages.map(msg => (
              <ChatMessage key={msg.id} msg={msg} />
            ))}
            {isThinking && <ThinkingIndicator />}
            <div ref={bottomRef} />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="error-toast">⚠ {error}</div>
        )}

        {/* Input */}
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask Dhruv anything..."
              rows={1}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
            >
              {loading ? '...' : 'Send'}
            </button>
          </div>
          <p className="input-hint">↵ enter to send · shift+↵ for new line</p>
        </div>
      </div>
    </div>
  )
}