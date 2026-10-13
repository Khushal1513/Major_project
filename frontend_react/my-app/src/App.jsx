import { useState, useRef, useEffect, useCallback } from 'react'
import { v4 as uuid } from 'uuid'
import Sidebar from './components/Sidebar'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import QuickPrompts from './components/QuickPrompts'
import DocumentPanel from './components/DocumentPanel'
import UploadDropzone from './components/UploadDropzone'
import { streamChatQuery, uploadDocument, streamDocQuestion } from './services/api'
import { useVoice } from './hooks/useVoice'
import './App.css'

const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

export default function App() {
  /* ── tabs & sidebar ────────────────────────────────────────── */
  const [activeTab, setActiveTab]       = useState('chat')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [topK, setTopK]                 = useState(5)

  /* ── chat state ────────────────────────────────────────────── */
  const [chatMessages, setChatMessages] = useState([
    {
      id: uuid(), role: 'assistant', type: 'info',
      content: 'Namaste! 🙏 I am your **Indian Laws AI Assistant**, specialised in the **Constitution of India**, **IPC / BNS**, **CrPC / BNSS**, and all major Indian statutes.\n\nAsk me anything about Indian law, or switch to **Document Analysis** to upload a court judgment, notice, or contract for instant AI-powered analysis.',
      time: now(),
    }
  ])
  const [chatStreaming, setChatStreaming] = useState(false)

  /* ── doc state ─────────────────────────────────────────────── */
  const [docMessages, setDocMessages]   = useState([])
  const [docStreaming, setDocStreaming]  = useState(false)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [isUploading, setIsUploading]   = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [docResult, setDocResult]       = useState(null)
  const [documentId, setDocumentId]     = useState(null)

  /* ── voice ─────────────────────────────────────────────────── */
  const [speakingMsgId, setSpeakingMsgId] = useState(null)
  const onTranscript = useCallback((t) => {
    if (activeTab === 'chat') handleChatSend(t)
    else handleDocSend(t)
  }, [activeTab]) // eslint-disable-line

  const { isListening, startListening, stopListening, speak, stopSpeaking, isSpeaking, isSupported: voiceSupported } =
    useVoice({ onTranscript })

  /* ── scroll refs ───────────────────────────────────────────── */
  const chatEndRef = useRef(null)
  const docEndRef  = useRef(null)

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMessages])
  useEffect(() => { docEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [docMessages])

  /* ══════════════════════════════════════════════════════════════
      CHAT TAB — send message
  ══════════════════════════════════════════════════════════════ */
  const handleChatSend = useCallback(async (text) => {
    if (chatStreaming) return

    const userMsg = { id: uuid(), role: 'user', content: text, time: now() }
    const aiId = uuid()
    const aiMsg = { id: aiId, role: 'assistant', content: '', streaming: true, time: now() }

    setChatMessages(prev => [...prev, userMsg, aiMsg])
    setChatStreaming(true)

    let fullText = ''
    let contexts = []

    await streamChatQuery({
      query: text,
      topK,
      onToken: (delta) => {
        fullText += delta
        setChatMessages(prev => prev.map(m => m.id === aiId ? { ...m, content: fullText } : m))
      },
      onContexts: (ctx) => { contexts = ctx },
      onDone: () => {
        setChatMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, content: fullText, streaming: false, contexts } : m
        ))
        setChatStreaming(false)
      },
      onError: (err) => {
        setChatMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, content: err, streaming: false, type: 'error' } : m
        ))
        setChatStreaming(false)
      },
    })
  }, [chatStreaming, topK])

  /* ══════════════════════════════════════════════════════════════
      DOC TAB — file upload
  ══════════════════════════════════════════════════════════════ */
  const handleFileSelect = useCallback(async (file) => {
    setUploadedFile(file)
    setIsUploading(true)
    setUploadProgress(0)
    setDocResult(null)
    setDocumentId(null)
    setDocMessages([])

    try {
      const result = await uploadDocument(file, setUploadProgress)
      setDocResult(result)
      setDocumentId(result.document_id)
      setDocMessages([{
        id: uuid(), role: 'assistant',
        content: `✅ **Document processed successfully!**\n\nI've analysed **${file.name}** and extracted the key legal information. You can see the full analysis above, and ask me any specific questions about this document below.`,
        time: now(),
      }])
    } catch (err) {
      setDocMessages([{
        id: uuid(), role: 'assistant', type: 'error',
        content: `Failed to process document: ${err.message}`,
        time: now(),
      }])
    } finally {
      setIsUploading(false)
    }
  }, [])

  /* ══════════════════════════════════════════════════════════════
      DOC TAB — Q&A
  ══════════════════════════════════════════════════════════════ */
  const handleDocSend = useCallback(async (text) => {
    if (docStreaming || !documentId) return

    const userMsg = { id: uuid(), role: 'user', content: text, time: now() }
    const aiId = uuid()
    const aiMsg = { id: aiId, role: 'assistant', content: '', streaming: true, time: now() }

    setDocMessages(prev => [...prev, userMsg, aiMsg])
    setDocStreaming(true)

    let fullText = ''
    let sources = []

    await streamDocQuestion({
      documentId,
      question: text,
      onToken: (delta) => {
        fullText += delta
        setDocMessages(prev => prev.map(m => m.id === aiId ? { ...m, content: fullText } : m))
      },
      onSources: (src) => { sources = src },
      onDone: () => {
        setDocMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, content: fullText, streaming: false, sources } : m
        ))
        setDocStreaming(false)
      },
      onError: (err) => {
        setDocMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, content: err, streaming: false, type: 'error' } : m
        ))
        setDocStreaming(false)
      },
    })
  }, [docStreaming, documentId])

  /* ── TTS ────────────────────────────────────────────────────── */
  const handleSpeak = useCallback((content, msgId) => {
    if (isSpeaking && speakingMsgId === msgId) {
      stopSpeaking()
      setSpeakingMsgId(null)
    } else {
      const plain = content.replace(/[*_`#>]/g, '').replace(/\n+/g, ' ')
      speak(plain)
      setSpeakingMsgId(msgId)
    }
  }, [isSpeaking, speakingMsgId, speak, stopSpeaking])

  /* ── clear all ──────────────────────────────────────────────── */
  const handleClear = () => {
    setChatMessages([])
    setDocMessages([])
    setDocResult(null)
    setDocumentId(null)
    setUploadedFile(null)
    stopSpeaking()
  }

  /* ══════════════════════════════════════════════════════════════
      RENDER
  ══════════════════════════════════════════════════════════════ */
  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--navy-950)' }}>
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        topK={topK}
        setTopK={setTopK}
        onClear={handleClear}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
      />

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        {/* ── Header ───────────────────────────────────────────── */}
        <header style={{
          padding: '14px 24px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--surface-2)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          flexShrink: 0,
        }}>
          <div>
            <h1 style={{ fontSize: 18, fontFamily: "'Crimson Pro', serif", fontWeight: 600, color: 'var(--gold-400)', margin: 0, lineHeight: 1.2 }}>
              {activeTab === 'chat' ? '⚖️ Law Q&A' : '📄 Document Analysis'}
            </h1>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '2px 0 0' }}>
              {activeTab === 'chat'
                ? 'Ask anything about IPC, BNS, Constitution, CrPC, BNSS & more'
                : 'Upload court judgments, notices, contracts for AI-powered legal analysis'}
            </p>
          </div>
          {/* Tab switch */}
          <div style={{ display: 'flex', gap: 4, background: 'var(--navy-800)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: 4 }}>
            {[{ id: 'chat', label: '💬 Q&A' }, { id: 'doc', label: '📄 Docs' }].map(({ id, label }) => (
              <button key={id} onClick={() => setActiveTab(id)} style={{
                padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none', cursor: 'pointer', fontSize: 13,
                background: activeTab === id ? 'var(--navy-600)' : 'transparent',
                color: activeTab === id ? 'var(--gold-400)' : 'var(--text-muted)',
                fontWeight: activeTab === id ? 600 : 400,
                transition: 'all 0.15s',
              }}>{label}</button>
            ))}
          </div>
        </header>

        {/* ── Chat Tab ─────────────────────────────────────────── */}
        {activeTab === 'chat' && (
          <>
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              {chatMessages.map(msg => (
                <ChatMessage
                  key={msg.id}
                  msg={msg}
                  onSpeak={(content) => handleSpeak(content, msg.id)}
                  isSpeaking={isSpeaking && speakingMsgId === msg.id}
                />
              ))}
              {chatStreaming && (
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '8px 0', color: 'var(--text-muted)', fontSize: 13 }}>
                  {[0,1,2].map(i => (
                    <span key={i} className="pulse" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--gold-400)', display: 'inline-block', animationDelay: `${i * 0.2}s` }}/>
                  ))}
                  <span style={{ marginLeft: 4 }}>Thinking…</span>
                </div>
              )}
              <div ref={chatEndRef}/>
            </div>

            <QuickPrompts mode="law" onSelect={handleChatSend}/>

            <ChatInput
              onSend={handleChatSend}
              onFileSelect={(file) => { setActiveTab('doc'); handleFileSelect(file) }}
              disabled={chatStreaming}
              placeholder="Ask about Indian law… e.g. What is Section 302 IPC?"
              isListening={isListening}
              onStartVoice={startListening}
              onStopVoice={stopListening}
              isSpeaking={isSpeaking}
              onStopSpeaking={stopSpeaking}
              voiceSupported={voiceSupported}
            />
          </>
        )}

        {/* ── Document Tab ──────────────────────────────────────── */}
        {activeTab === 'doc' && (
          <>
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14 }}>

              {/* Upload zone always visible at top */}
              <UploadDropzone
                onFile={handleFileSelect}
                isUploading={isUploading}
                uploadProgress={uploadProgress}
                uploadDone={!!docResult && !isUploading}
              />

              {/* Analysis results */}
              {docResult && !isUploading && (
                <DocumentPanel result={docResult}/>
              )}

              {/* Doc chat messages */}
              {docMessages.length > 0 && (
                <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.8 }}>
                    💬 Document Q&A
                  </div>
                  {docMessages.map(msg => (
                    <ChatMessage
                      key={msg.id}
                      msg={msg}
                      onSpeak={(content) => handleSpeak(content, msg.id)}
                      isSpeaking={isSpeaking && speakingMsgId === msg.id}
                    />
                  ))}
                  {docStreaming && (
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '4px 0', color: 'var(--text-muted)', fontSize: 13 }}>
                      {[0,1,2].map(i => (
                        <span key={i} className="pulse" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--gold-400)', display: 'inline-block', animationDelay: `${i * 0.2}s` }}/>
                      ))}
                      <span style={{ marginLeft: 4 }}>Analysing…</span>
                    </div>
                  )}
                </div>
              )}

              {/* Empty state */}
              {!docResult && !isUploading && (
                <div style={{ textAlign: 'center', padding: '20px 0' }}>
                  <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                    Upload a PDF or image to get started.<br/>
                    <span style={{ color: 'var(--text-secondary)' }}>Supported: court judgments, government notices, legal contracts, FIR copies, scanned documents.</span>
                  </p>
                </div>
              )}

              <div ref={docEndRef}/>
            </div>

            {/* Quick prompts for doc mode only when doc uploaded */}
            {documentId && <QuickPrompts mode="doc" onSelect={handleDocSend}/>}

            <ChatInput
              onSend={documentId ? handleDocSend : () => {}}
              onFileSelect={handleFileSelect}
              disabled={docStreaming || isUploading}
              placeholder={documentId ? 'Ask about this document…' : 'Upload a document first, then ask questions…'}
              isListening={isListening}
              onStartVoice={startListening}
              onStopVoice={stopListening}
              isSpeaking={isSpeaking}
              onStopSpeaking={stopSpeaking}
              voiceSupported={voiceSupported}
              activeFile={isUploading ? uploadedFile : null}
              isUploading={isUploading}
            />
          </>
        )}
      </div>
    </div>
  )
}