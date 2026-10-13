import React, { useRef, useState } from 'react'
import { Upload, FileText, Image, Loader, CheckCircle } from 'lucide-react'

export default function UploadDropzone({ onFile, isUploading, uploadProgress, uploadDone }) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handle = (file) => {
    if (!file) return
    const allowed = ['application/pdf','image/jpeg','image/png','image/tiff','image/webp','image/jpg']
    if (!allowed.includes(file.type)) { alert('Only PDF or image files are supported.'); return }
    if (file.size > 20 * 1024 * 1024) { alert('File must be under 20 MB.'); return }
    onFile(file)
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={e => { e.preventDefault(); setDragOver(false); handle(e.dataTransfer.files[0]) }}
      onClick={() => !isUploading && inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragOver ? 'var(--gold-500)' : uploadDone ? 'var(--emerald-400)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-lg)',
        padding: '32px 20px',
        textAlign: 'center',
        cursor: isUploading ? 'default' : 'pointer',
        background: dragOver ? 'rgba(201,162,39,0.04)' : uploadDone ? 'rgba(52,211,153,0.04)' : 'var(--navy-800)',
        transition: 'all 0.2s',
        userSelect: 'none',
      }}
    >
      <input ref={inputRef} type="file" style={{ display: 'none' }}
        accept=".pdf,.jpg,.jpeg,.png,.tiff,.webp"
        onChange={e => handle(e.target.files?.[0])}/>

      {isUploading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
          <Loader size={32} className="spin" color="var(--gold-400)"/>
          <p style={{ color: 'var(--gold-300)', fontSize: 14, fontWeight: 500 }}>Processing document…</p>
          <div style={{ width: 200, height: 4, background: 'var(--navy-700)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${uploadProgress}%`, background: 'var(--gold-500)', borderRadius: 4, transition: 'width 0.3s' }}/>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>{uploadProgress}%</p>
        </div>
      ) : uploadDone ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <CheckCircle size={32} color="var(--emerald-400)"/>
          <p style={{ color: 'var(--emerald-400)', fontSize: 14, fontWeight: 500 }}>Document processed!</p>
          <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Click to upload a different file</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <Upload size={32} color={dragOver ? 'var(--gold-400)' : 'var(--text-muted)'}/>
          <p style={{ color: dragOver ? 'var(--gold-300)' : 'var(--text-secondary)', fontSize: 14, fontWeight: 500 }}>
            {dragOver ? 'Drop to upload' : 'Drag & drop or click to upload'}
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>PDF, JPG, PNG, TIFF, WEBP · Max 20 MB</p>
          <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
            {[{ icon: FileText, label: 'Court judgments' }, { icon: Image, label: 'Scanned docs' }].map(({ icon: Icon, label }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                <Icon size={13}/> {label}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}