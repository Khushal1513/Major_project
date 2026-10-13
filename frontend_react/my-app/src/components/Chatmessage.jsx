import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Scale, User, ChevronDown, ChevronUp, Volume2, VolumeX, BookOpen, FileText } from 'lucide-react'

function SourcesPanel({ contexts, sources }) {
  const [open, setOpen] = useState(false)
  const items = contexts || sources || []
  if (!items.length) return null

  return (
    <div style={{ marginTop: 10 }}>
      <button onClick={() => setOpen(!open)} style={{
        display: 'flex', alignItems: 'center', gap: 6,
        background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)', padding: '5px 12px',
        color: 'var(--text-secondary)', fontSize: 12, cursor: 'pointer',
        transition: 'background 0.15s',
      }}>
        <BookOpen size={12}/>
        {open ? 'Hide' : 'View'} {items.length} source{items.length !== 1 ? 's' : ''}
        {open ? <ChevronUp size={12}/> : <ChevronDown size={12}/>}
      </button>
      {open && (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8 }} className="slide-up">
          {items.map((item, i) => {
            const isDoc = item.source === 'document'
            const score = item.score ?? item.relevance_score ?? 0
            const text = item.text ?? item.text_preview ?? ''
            const label = isDoc ? 'Uploaded Document' : (item.source || 'Law Knowledge Base')
            const from = item.section ? `${item.source} – ${item.section}` : label
            return (
              <div key={i} style={{
                background: 'var(--navy-800)', border: '1px solid var(--border)',
                borderLeft: `3px solid ${isDoc ? '#7b1fa2' : 'var(--navy-400)'}`,
                borderRadius: 'var(--radius-sm)', padding: '10px 14px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: isDoc ? '#ce93d8' : 'var(--navy-200)' }}>
                    {isDoc ? <FileText size={12} style={{ display: 'inline', marginRight: 4 }}/> : null}
                    {from}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {score > 0 ? `score: ${score.toFixed ? score.toFixed(3) : score}` : ''}
                  </span>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                  {text.slice(0, 300)}{text.length > 300 ? '…' : ''}
                </p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default function ChatMessage({ msg, onSpeak, isSpeaking }) {
  const isUser = msg.role === 'user'
  const isError = msg.type === 'error'

  const bgColor = isUser
    ? 'rgba(37,53,112,0.6)'
    : isError
      ? 'rgba(220,38,38,0.08)'
      : 'rgba(26,37,64,0.7)'

  const borderColor = isUser
    ? 'rgba(74,111,165,0.4)'
    : isError
      ? 'rgba(220,38,38,0.3)'
      : 'rgba(201,162,39,0.15)'

  return (
    <div className="slide-up" style={{
      display: 'flex', gap: 12, padding: '4px 0',
      flexDirection: isUser ? 'row-reverse' : 'row',
    }}>
      {/* Avatar */}
      <div style={{
        width: 34, height: 34, borderRadius: '50%', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: isUser ? 'var(--navy-600)' : 'rgba(201,162,39,0.15)',
        border: `1px solid ${isUser ? 'var(--navy-400)' : 'rgba(201,162,39,0.3)'}`,
        marginTop: 2,
      }}>
        {isUser ? <User size={15} color="var(--navy-200)"/> : <Scale size={15} color="var(--gold-400)"/>}
      </div>

      {/* Bubble */}
      <div style={{
        maxWidth: '80%', minWidth: 60,
        background: bgColor,
        border: `1px solid ${borderColor}`,
        borderRadius: isUser ? 'var(--radius-lg) var(--radius-sm) var(--radius-lg) var(--radius-lg)' : 'var(--radius-sm) var(--radius-lg) var(--radius-lg) var(--radius-lg)',
        padding: '12px 16px',
        boxShadow: 'var(--shadow-sm)',
      }}>
        {/* Header for AI */}
        {!isUser && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--gold-400)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.8 }}>
              {isError ? '⚠ Notice' : '⚖ AI Assistant'}
            </span>
            {msg.content && !msg.streaming && !isError && (
              <button onClick={() => onSpeak(msg.content)} style={{
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: isSpeaking ? 'var(--gold-400)' : 'var(--text-muted)',
                padding: 4, borderRadius: 4, display: 'flex', alignItems: 'center',
                transition: 'color 0.15s',
              }} title={isSpeaking ? 'Stop speaking' : 'Read aloud'}>
                {isSpeaking ? <VolumeX size={13}/> : <Volume2 size={13}/>}
              </button>
            )}
          </div>
        )}

        {/* Content */}
        <div style={{ fontSize: 14.5, lineHeight: 1.7, color: isError ? '#fca5a5' : 'var(--text-primary)' }}>
          {msg.streaming ? (
            <span>
              <ReactMarkdown className="md-content" remarkPlugins={[remarkGfm]}>
                {msg.content || ''}
              </ReactMarkdown>
              <span className="cursor"/>
            </span>
          ) : (
            <ReactMarkdown className="md-content" remarkPlugins={[remarkGfm]}>
              {msg.content || ''}
            </ReactMarkdown>
          )}
        </div>

        {/* Sources */}
        {!isUser && !msg.streaming && (
          <SourcesPanel contexts={msg.contexts} sources={msg.sources}/>
        )}

        {/* Timestamp */}
        <div style={{ marginTop: 6, fontSize: 10.5, color: 'var(--text-muted)', textAlign: isUser ? 'right' : 'left' }}>
          {msg.time}
        </div>
      </div>
    </div>
  )
}