import React from 'react'

const LAW_PROMPTS = [
  'What is Section 302 IPC?',
  'Explain Article 21 of the Constitution',
  'What are Fundamental Rights?',
  'Difference between IPC and BNS',
  'What is CrPC?',
  'Explain Section 377 IPC',
  'What is Article 14?',
  'Bail provisions under BNS',
]

const DOC_PROMPTS = [
  'Summarize this document in detail',
  'What are the most important legal points?',
  'Which constitutional articles apply?',
  'Does this affect any fundamental rights?',
  'What legal actions are mentioned?',
  'Explain key clauses in simple language',
]

export default function QuickPrompts({ mode = 'law', onSelect }) {
  const prompts = mode === 'doc' ? DOC_PROMPTS : LAW_PROMPTS

  return (
    <div style={{ padding: '0 16px 12px' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.8 }}>
        Quick questions
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {prompts.map((p) => (
          <button key={p} onClick={() => onSelect(p)} style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid var(--border)',
            borderRadius: 20,
            padding: '5px 12px',
            fontSize: 12,
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            transition: 'all 0.15s',
            whiteSpace: 'nowrap',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'rgba(201,162,39,0.08)'
            e.currentTarget.style.borderColor = 'rgba(201,162,39,0.3)'
            e.currentTarget.style.color = 'var(--gold-300)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
            e.currentTarget.style.borderColor = 'var(--border)'
            e.currentTarget.style.color = 'var(--text-secondary)'
          }}>
            {p}
          </button>
        ))}
      </div>
    </div>
  )
}