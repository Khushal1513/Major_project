// api.js

const BASE = "http://127.0.0.1:8000";

/**
 * Ask a law-related question.
 */
export async function streamChatQuery({
  query,
  topK = 5,
  onToken,
  onContexts,
  onDone,
  onError,
}) {
  try {
    const res = await fetch(`${BASE}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        top_k: topK,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();

    if (!data.success) {
      onError?.(data.answer || "Request failed");
      return;
    }

    if (data.sources) {
      onContexts?.(data.sources);
    }

    if (data.answer) {
      onToken?.(data.answer);
    }

    onDone?.();
  } catch (err) {
    console.error(err);
    onError?.(err.message);
  }
}

/**
 * Upload a document.
 */
export async function uploadDocument(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.open("POST", `${BASE}/upload`);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        onProgress?.(percent);
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid JSON response"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || "Upload failed"));
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      }
    };

    xhr.onerror = () => {
      reject(new Error("Network error"));
    };

    xhr.send(formData);
  });
}

/**
 * Ask a question about an uploaded document.
 */
export async function streamDocQuestion({
  documentId,
  question,
  onToken,
  onSources,
  onDone,
  onError,
}) {
  try {
    const res = await fetch(`${BASE}/document-chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        document_id: documentId,
        question,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();

    if (data.answer) {
      onToken?.(data.answer);
    }

    if (data.sources) {
      onSources?.(data.sources);
    }

    onDone?.();
  } catch (err) {
    console.error(err);
    onError?.(err.message);
  }
}