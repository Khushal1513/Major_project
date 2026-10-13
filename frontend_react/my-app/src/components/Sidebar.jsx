import React from 'react'
import { Scale, MessageSquare, FileText, Trash2, ChevronLeft, ChevronRight, Info } from 'lucide-react'

const AshokaSVG = () => (
  <svg viewBox="0 0 120 120" width="64" height="64" xmlns="http://www.w3.org/2000/svg">
    <circle cx="60" cy="60" r="54" fill="none" stroke="#c9a227" strokeWidth="2" opacity="0.4"/>
    <circle cx="60" cy="60" r="36" fill="none" stroke="#c9a227" strokeWidth="1.5" opacity="0.7"/>
    <circle cx="60" cy="60" r="6" fill="#c9a227"/>
    {Array.from({ length: 24 }, (_, i) => (
      <line key={i} x1="60" y1="54" x2="60" y2="25"
        stroke="#c9a227" strokeWidth="1.2" opacity="0.8"
        transform={`rotate(${i * 15} 60 60)`}/>
    ))}
  </svg>
)

export default function Sidebar({ activeTab, setActiveTab, topK, setTopK, onClear, collapsed, setCollapsed }) {
  return (
    <aside style={{
      width: collapsed ? 60 : 260,
      minWidth: collapsed ? 60 : 260,
      background: 'var(--surface-2)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      transition: 'width 0.25s ease, min-width 0.25s ease',
      overflow: 'hidden',
      position: 'relative',
      zIndex: 10,
    }}>
      {/* Toggle */}
      <button onClick={() => setCollapsed(!collapsed)} style={{
        position: 'absolute', top: 12, right: -14, zIndex: 20,
        background: 'var(--navy-700)', border: '1px solid var(--border)',
        color: 'var(--text-secondary)', borderRadius: '50%',
        width: 28, height: 28, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: 'var(--shadow-sm)',
      }}>
        {collapsed ? <ChevronRight size={14}/> : <ChevronLeft size={14}/>}
      </button>

      {/* Header */}
      <div style={{ padding: collapsed ? '20px 0' : '24px 20px', display: 'flex', flexDirection: 'column', alignItems: collapsed ? 'center' : 'flex-start', gap: 12, borderBottom: '1px solid var(--border)' }}>
        <AshokaSVG />
        {!collapsed && (
          <div>
            <div style={{ fontFamily: "'Crimson Pro', serif", fontSize: 20, fontWeight: 600, color: 'var(--gold-400)', letterSpacing: 0.3 }}>Indian Laws AI</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>Constitution & Legal Assistant</div>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav style={{ padding: collapsed ? '12px 0' : '12px 12px', flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {[
          { id: 'chat', icon: MessageSquare, label: 'Law Q&A' },
          { id: 'doc', icon: FileText, label: 'Document Analysis' },
        ].map(({ id, icon: Icon, label }) => (
          <button key={id} onClick={() => setActiveTab(id)} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: collapsed ? '10px 0' : '10px 12px',
            justifyContent: collapsed ? 'center' : 'flex-start',
            background: activeTab === id ? 'rgba(201,162,39,0.12)' : 'transparent',
            border: activeTab === id ? '1px solid rgba(201,162,39,0.25)' : '1px solid transparent',
            borderRadius: 'var(--radius-sm)', cursor: 'pointer',
            color: activeTab === id ? 'var(--gold-400)' : 'var(--text-secondary)',
            fontSize: 14, fontWeight: activeTab === id ? 600 : 400,
            transition: 'all 0.15s',
          }}>
            <Icon size={17} style={{ flexShrink: 0 }}/>
            {!collapsed && label}
          </button>
        ))}

        {!collapsed && (
          <div style={{ marginTop: 16, padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.8 }}>Sources to retrieve</div>
            <input type="range" min={3} max={10} value={topK} onChange={e => setTopK(+e.target.value)}
              style={{ width: '100%', accentColor: 'var(--gold-500)', cursor: 'pointer' }}/>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              <span>3</span><span style={{ color: 'var(--gold-400)', fontWeight: 600 }}>{topK}</span><span>10</span>
            </div>
          </div>
        )}
      </nav>

      {/* Bottom */}
      {!collapsed && (
        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
          <button onClick={onClear} style={{
            display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
            background: 'transparent', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            color: 'var(--text-secondary)', fontSize: 13, cursor: 'pointer', width: '100%', transition: 'all 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--red-600)'}
          onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}>
            <Trash2 size={14}/> Clear History
          </button>
          <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
            <Info size={12} style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: 2 }}/>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
              For informational purposes only. Consult a qualified lawyer for legal advice.
            </p>
          </div>
        </div>
      )}
    </aside>
  )
}