import React, { useRef, useState, useCallback } from 'react'
import { Mic, MicOff, Send, Paperclip, X, Image, Volume2, VolumeX, FileText, Loader } from 'lucide-react'

export default function ChatInput({
  onSend,
  onFileSelect,
  disabled,
  placeholder,
  isListening,
  onStartVoice,
  onStopVoice,
  isSpeaking,
  onStopSpeaking,
  voiceSupported,
  activeFile,
  onClearFile,
  isUploading,
}) {
  const [text, setText] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
    textareaRef.current?.focus()
  }, [text, disabled, onSend])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleFile = (file) => {
    if (!file) return
    const allowed = ['application/pdf','image/jpeg','image/png','image/tiff','image/webp','image/jpg']
    if (!allowed.includes(file.type)) {
      alert('Only PDF, JPG, PNG, TIFF, WEBP files are supported.')
      return
    }
    if (file.size > 20 * 1024 * 1024) {
      alert('File must be under 20 MB.')
      return
    }
    onFileSelect(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const isPDF = activeFile?.type === 'application/pdf'

  return (
    <div style={{ padding: '12px 16px 16px', borderTop: '1px solid var(--border)', background: 'var(--surface-2)' }}>
      {/* Active file badge */}
      {activeFile && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10,
          padding: '7px 12px', background: 'rgba(123,31,162,0.1)',
          border: '1px solid rgba(123,31,162,0.3)', borderRadius: 'var(--radius-sm)',
        }}>
          {isPDF ? <FileText size={14} color="#ce93d8"/> : <Image size={14} color="#ce93d8"/>}
          <span style={{ fontSize: 13, color: '#ce93d8', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {activeFile.name}
          </span>
          {isUploading && <Loader size={13} className="spin" color="#ce93d8"/>}
          {!isUploading && (
            <button onClick={onClearFile} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ce93d8', display: 'flex', padding: 2 }}>
              <X size={13}/>
            </button>
          )}
        </div>
      )}

      {/* Drag overlay */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        style={{
          display: 'flex', gap: 8, alignItems: 'flex-end',
          background: dragOver ? 'rgba(201,162,39,0.06)' : 'var(--navy-800)',
          border: `1.5px solid ${dragOver ? 'var(--gold-500)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '8px 8px 8px 14px',
          transition: 'border-color 0.15s, background 0.15s',
        }}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKey}
          placeholder={isListening ? '🎙 Listening…' : placeholder}
          disabled={disabled}
          rows={1}
          style={{
            flex: 1, background: 'transparent', border: 'none', outline: 'none',
            color: isListening ? 'var(--gold-300)' : 'var(--text-primary)',
            fontSize: 14.5, lineHeight: 1.55, resize: 'none',
            fontFamily: 'inherit', minHeight: 24, maxHeight: 160,
            overflow: 'auto', paddingTop: 4, paddingBottom: 4,
          }}
          onInput={e => {
            e.target.style.height = 'auto'
            e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
          }}
        />

        {/* Attachment */}
        <input ref={fileInputRef} type="file"
          accept=".pdf,.jpg,.jpeg,.png,.tiff,.webp"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files?.[0])}/>

        <IconBtn onClick={() => fileInputRef.current?.click()} title="Attach PDF or image" disabled={disabled}>
          <Paperclip size={17}/>
        </IconBtn>

        {/* Voice */}
        {voiceSupported && (
          <IconBtn
            onClick={isListening ? onStopVoice : onStartVoice}
            title={isListening ? 'Stop recording' : 'Voice input'}
            active={isListening}
            disabled={disabled}
          >
            {isListening
              ? <MicOff size={17} color="var(--gold-400)"/>
              : <Mic size={17}/>}
          </IconBtn>
        )}

        {/* TTS toggle */}
        {isSpeaking && (
          <IconBtn onClick={onStopSpeaking} title="Stop speaking" active>
            <VolumeX size={17} color="var(--gold-400)"/>
          </IconBtn>
        )}

        {/* Send */}
        <button
          onClick={handleSubmit}
          disabled={!text.trim() || disabled}
          title="Send (Enter)"
          style={{
            width: 36, height: 36, borderRadius: '50%', border: 'none', cursor: 'pointer',
            background: !text.trim() || disabled ? 'rgba(201,162,39,0.15)' : 'var(--gold-500)',
            color: !text.trim() || disabled ? 'var(--text-muted)' : 'var(--navy-950)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0, transition: 'all 0.15s',
          }}
        >
          <Send size={16}/>
        </button>
      </div>

      <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
        Enter to send · Shift+Enter for new line · Drag & drop files supported
      </div>
    </div>
  )
}

function IconBtn({ onClick, title, active, disabled, children }) {
  return (
    <button onClick={onClick} title={title} disabled={disabled} style={{
      width: 34, height: 34, borderRadius: 'var(--radius-sm)',
      background: active ? 'rgba(201,162,39,0.12)' : 'transparent',
      border: `1px solid ${active ? 'rgba(201,162,39,0.3)' : 'transparent'}`,
      color: active ? 'var(--gold-400)' : 'var(--text-secondary)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexShrink: 0, opacity: disabled ? 0.4 : 1,
      transition: 'all 0.15s',
    }}>
      {children}
    </button>
  )
}