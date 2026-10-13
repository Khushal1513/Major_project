import { useState, useRef, useCallback, useEffect } from 'react'

/**
 * useVoice – microphone input (STT) + text-to-speech (TTS)
 *
 * Returns:
 *  isListening, transcript, startListening, stopListening,
 *  isSpeaking, speak, stopSpeaking, isSupported
 */
export function useVoice({ onTranscript } = {}) {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [isSpeaking, setIsSpeaking] = useState(false)
  const recognitionRef = useRef(null)
  const utteranceRef = useRef(null)

  const isSupported = typeof window !== 'undefined' &&
    ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)

  const startListening = useCallback(() => {
    if (!isSupported) return
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    const rec = new SR()
    rec.lang = 'en-IN'
    rec.continuous = false
    rec.interimResults = true
    rec.maxAlternatives = 1

    rec.onstart = () => setIsListening(true)
    rec.onend = () => setIsListening(false)

    rec.onresult = (e) => {
      let interim = ''
      let final = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) final += t
        else interim += t
      }
      const text = final || interim
      setTranscript(text)
      if (final) onTranscript?.(final)
    }

    rec.onerror = () => setIsListening(false)

    recognitionRef.current = rec
    rec.start()
  }, [isSupported, onTranscript])

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [])

  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(text)
    utt.lang = 'en-IN'
    utt.rate = 0.95
    utt.pitch = 1
    utt.onstart = () => setIsSpeaking(true)
    utt.onend = () => setIsSpeaking(false)
    utt.onerror = () => setIsSpeaking(false)
    utteranceRef.current = utt
    window.speechSynthesis.speak(utt)
  }, [])

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel()
    setIsSpeaking(false)
  }, [])

  useEffect(() => () => {
    recognitionRef.current?.abort()
    window.speechSynthesis?.cancel()
  }, [])

  return { isListening, transcript, startListening, stopListening, isSpeaking, speak, stopSpeaking, isSupported }
}