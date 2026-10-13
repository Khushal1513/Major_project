import React, { useState } from 'react'
import { FileText, ChevronDown, ChevronUp, Shield, BookOpen, Scale, AlignLeft } from 'lucide-react'

export default function DocumentPanel({ result }) {
  const [textOpen, setTextOpen] = useState(false)

  if (!result) return null

  const Section = ({ icon: Icon, title, color = 'var(--gold-400)', children }) => (
    <div style={{ background: 'var(--navy-800)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '14px 16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
        <Icon size={14} color={color}/>
        <span style={{ fontSize: 12, fontWeight: 600, color, textTransform: 'uppercase', letterSpacing: 0.8 }}>{title}</span>
      </div>
      {children}
    </div>
  )

  const Tags = ({ items, color = 'var(--gold-300)', bg = 'rgba(201,162,39,0.1)', border = 'rgba(201,162,39,0.25)' }) => (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
      {(items || []).length === 0
        ? <span style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>None identified</span>
        : items.map((item, i) => (
          <span key={i} style={{ display: 'inline-flex', alignItems: 'center', background: bg, color, border: `1px solid ${border}`, padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 500 }}>
            {item}
          </span>
        ))}
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '4px 0 12px' }}>
      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 10 }}>
        {[
          { label: 'Pages', value: result.total_pages },
          { label: 'Chunks', value: result.total_chunks },
          { label: 'Doc ID', value: result.document_id?.slice(0, 8) + '…' },
        ].map(({ label, value }) => (
          <div key={label} style={{ flex: 1, background: 'var(--navy-800)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', textAlign: 'center' }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--gold-400)', fontFamily: "'Crimson Pro', serif" }}>{value}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <Section icon={AlignLeft} title="Summary" color="var(--navy-200)">
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.65, margin: 0 }}>{result.summary}</p>
      </Section>

      {/* Key Legal Points */}
      <Section icon={Scale} title="Key Legal Points" color="var(--gold-400)">
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {(result.key_legal_points || []).map((pt, i) => (
            <li key={i} style={{ display: 'flex', gap: 8, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              <span style={{ color: 'var(--gold-500)', flexShrink: 0, marginTop: 1 }}>•</span> {pt}
            </li>
          ))}
        </ul>
      </Section>

      {/* Rights & Articles */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <Section icon={Shield} title="Fundamental Rights" color="#ce93d8">
          <Tags items={result.fundamental_rights_involved} color="#ce93d8" bg="rgba(123,31,162,0.1)" border="rgba(123,31,162,0.25)"/>
        </Section>
        <Section icon={BookOpen} title="Constitutional Articles" color="var(--gold-400)">
          <Tags items={result.constitutional_articles}/>
        </Section>
      </div>

      {/* Legal References */}
      <Section icon={FileText} title="Legal References" color="var(--navy-200)">
        <Tags items={result.legal_references} color="var(--navy-200)" bg="rgba(74,111,165,0.1)" border="rgba(74,111,165,0.25)"/>
      </Section>

      {/* Extracted text preview */}
      <div style={{ background: 'var(--navy-800)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
        <button onClick={() => setTextOpen(!textOpen)} style={{
          width: '100%', padding: '10px 16px', background: 'transparent', border: 'none', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          color: 'var(--text-secondary)', fontSize: 12, fontWeight: 600,
        }}>
          <span>📄 Extracted Text Preview</span>
          {textOpen ? <ChevronUp size={13}/> : <ChevronDown size={13}/>}
        </button>
        {textOpen && (
          <div style={{ padding: '0 16px 14px', borderTop: '1px solid var(--border)' }}>
            <pre style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: '12px 0 0', maxHeight: 180, overflowY: 'auto' }}>
              {result.extracted_text_preview}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}