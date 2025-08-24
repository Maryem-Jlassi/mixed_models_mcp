// context/Context.jsx
import React from 'react';
import { createContext, useState, useCallback, useEffect } from "react";

export const Context = createContext();

const ContextProvider = (props) => {
  // UI State
  const [input, setInput] = useState("");
  const [recentPrompt, setRecentPrompt] = useState("");
  const [prevPrompts, setPrevPrompts] = useState([]);
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState("");
  
  // Application State
  const [file, setFile] = useState(null);
  const [sessionId, setSessionId] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [currentStatus, setCurrentStatus] = useState('');
  const [statusHistory, setStatusHistory] = useState([]);
  const [streamingText, setStreamingText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  // Current workflow (general, cv_only, job_only, cv_job_matching)
  const [workflow, setWorkflow] = useState('general');
  // Structured progress from backend
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressStage, setProgressStage] = useState('');
  const [progressMessage, setProgressMessage] = useState('');
  const [toolSteps, setToolSteps] = useState([]); // {tool, success, atPercent}
  // Sessions state
  const [sessions, setSessions] = useState([]);
  // Streamed jobs state
  const [jobs, setJobs] = useState([]);
  // Full conversation for current session
  const [conversation, setConversation] = useState([]);
  // Deep crawl activity indicator
  const [deepCrawlActive, setDeepCrawlActive] = useState(false);
  // Base URL of the crawled site (derived from detected URL)
  const [crawledBaseUrl, setCrawledBaseUrl] = useState('');
  // Streamed crawled pages markdown previews
  const [pagesMarkdown, setPagesMarkdown] = useState([]);
  // Detailed thinking steps streamed from backend
  const [thinkingSteps, setThinkingSteps] = useState([]);
  // Provider selection: 'local' (Ollama) or 'groq'
  const [provider, setProvider] = useState('local');

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Helper: Authorization header if logged in
  const getAuthHeaders = () => {
    try {
      const token = localStorage.getItem('tm_token');
      return token ? { 'Authorization': `Bearer ${token}` } : {};
    } catch {
      return {};
    }
  };

  // Helper: is authenticated
  const isAuthenticated = () => {
    try {
      return Boolean(localStorage.getItem('tm_token'));
    } catch {
      return false;
    }
  };

  // Fetch sessions list
  const refreshSessions = useCallback(async () => {
    if (!isAuthenticated()) {
      setSessions([]);
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/sessions`, {
        headers: {
          ...getAuthHeaders(),
        },
      });
      if (res.status === 401) {
        // Token invalid/expired, clear it to stop further unauthorized calls
        try {
          localStorage.removeItem('tm_token');
          localStorage.removeItem('tm_user');
        } catch {}
        setSessions([]);
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (e) {
      // ignore
    }
  }, [API_BASE_URL]);


  // New chat function - starts fresh session
  const newChat = () => {
    setLoading(false);
    setShowResult(false);
    setResultData('');
    setRecentPrompt('');
    setCurrentStatus('');
    setStatusHistory([]);
    setStreamingText('');
    setIsStreaming(false);
    setFile(null);
    setSessionId(''); // Clear session to start fresh
    setUploadStatus('');
    setJobs([]);
    setConversation([]);
    setWorkflow('general');
  };

  const onSent = useCallback(async (prompt, file) => {
    if (!prompt && !file) return;

    setLoading(true);
    setShowResult(true);
    setResultData(""); // Clear previous result
    setStreamingText(""); // Clear streaming text
    setIsStreaming(false);
    setRecentPrompt(prompt || (file ? "Analyzing CV..." : ""));
    setInput("");
    setJobs([]); // reset jobs for new request
    setPagesMarkdown([]); // reset crawled pages previews for new request
    setThinkingSteps([]); // reset thinking steps for new request
    // Optimistic immediate feedback
    setProgressPercent(1);
    setProgressStage('planning');
    setProgressMessage('Starting...');
    setCurrentStatus('Planning...');

    // Immediately commit the user message so the chat view renders and progress is shown under it
    const userMsg = (prompt || (file ? "Analyzing CV..." : "")).toString();
    if (userMsg) {
      setConversation(prev => [...prev, { role: 'user', content: userMsg, timestamp: new Date().toISOString() }]);
    }

    if (file) {
      setIsUploading(true);
      setUploadStatus('Uploading and processing file...');
    }

    try {
      const formData = new FormData();
      // Detect job URL in the prompt to better route the backend
      const urlRegex = /(https?:\/\/[^\s]+)/i;
      const urlMatch = (prompt || '').match(urlRegex);
      const detectedUrl = urlMatch ? urlMatch[1] : '';
      // derive base/origin
      if (detectedUrl) {
        try {
          const u = new URL(detectedUrl);
          setCrawledBaseUrl(u.origin);
        } catch {}
      } else {
        setCrawledBaseUrl('');
      }

      // Choose workflow: CV upload -> cv_analysis, URL -> job_extraction, else general
      const workflowType = file ? 'cv_analysis' : (detectedUrl ? 'job_extraction' : 'general');
      formData.append('workflow_type', workflowType);
      formData.append('message', prompt || '');
      if (detectedUrl) {
        // Explicitly send job_url for backend routing
        formData.append('job_url', detectedUrl);
      }
      if (file) {
        formData.append('cv_file', file);
      }
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      // Set deep crawl indicator if we are sending a URL-based request without a file
      setDeepCrawlActive(Boolean(detectedUrl) && !file);

      const endpointPath = provider === 'groq' ? '/groq/workflow' : '/workflow';
      const response = await fetch(`${API_BASE_URL}${endpointPath}`, {
        method: 'POST',
        headers: {
          ...getAuthHeaders(),
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
        throw new Error(errorData.detail || 'Failed to get a response from the server.');
      }

      // We've already committed the user message above

      // Handle streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let result = '';
      let stopStream = false;
      let assistantAccum = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // Decode the chunk and add to buffer
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete JSON objects in the buffer
        let startIndex = 0;
        let jsonStart = buffer.indexOf('{', startIndex);
        
        while (jsonStart !== -1) {
          try {
            const jsonEnd = buffer.indexOf('\n', jsonStart);
            if (jsonEnd === -1) break; // Incomplete JSON object
            
            const jsonStr = buffer.substring(jsonStart, jsonEnd);
            const data = JSON.parse(jsonStr);
            
            // Handle different message types from enhanced orchestrator
            if (data.type === 'session' && data.session_id) {
              // Save session id immediately to avoid starting a new session on next send
              setSessionId(data.session_id);
            } else 
            if (data.type === 'progress') {
              // Structured progress update
              const pct = Number.isFinite(data.percent) ? Math.max(0, Math.min(100, Math.floor(data.percent))) : progressPercent;
              setProgressPercent(pct);
              if (data.stage) setProgressStage(String(data.stage));
              if (data.message) setProgressMessage(String(data.message));
              if (!isStreaming) setIsStreaming(true);
              // Reflect a concise status string for inline banner
              const stageLabel = (data.stage || '').toString().replace(/_/g, ' ');
              const statusMsg = data.message ? `${stageLabel ? stageLabel + ': ' : ''}${data.message}` : stageLabel;
              if (statusMsg) setCurrentStatus(statusMsg);
              // Keep status history compact
              setStatusHistory(prev => [...prev, {
                message: statusMsg || 'Working...',
                timestamp: Date.now(),
                type: 'info',
                percent: pct
              }]);
            } else 
            if (data.type === 'plan' && data.workflow) {
              setWorkflow(String(data.workflow)); // Update workflow type
            } else 
            if (data.type === 'final' && data.workflow) {
              setWorkflow(String(data.workflow)); // Update workflow type
            } else 
            if (data.type === 'response' && data.content) {
              result += data.content;
              assistantAccum += data.content;
              setIsStreaming(true);
              setStreamingText(prev => prev + data.content);
              if (process.env.NODE_ENV !== 'production') {
                try { console.log('[stream][response]', data.content.slice(0, 80)); } catch {}
              }
              // user message already committed
            } else if (data.type === 'status' && data.message) {
              // Enhanced status handling - show useful status messages dynamically
              const message = data.message;
              
              // Clean message by removing emojis for display
              const cleanMessage = message.replace(/[🔍📄✅❌🔗🎯]/g, '').trim();
              
              // Update current status for dynamic display
              setCurrentStatus(cleanMessage);
              
              // Add to status history
              setStatusHistory(prev => [...prev, {
                message: cleanMessage,
                timestamp: Date.now(),
                type: message.includes('✅') ? 'success' : 
                      message.includes('❌') ? 'error' : 
                      message.includes('🔍') || message.includes('📄') ? 'processing' : 'info'
              }]);
              
              // Don't show status in main result area anymore
            } else if (data.type === 'tool_complete') {
              const tool = data.tool || 'tool';
              const success = !!data.success;
              setToolSteps(prev => [...prev, { tool, success, atPercent: progressPercent, ts: Date.now() }]);
            } else if (data.type === 'result_summary' && data.data) {
              // Handle job results summary with better formatting
              const jobCount = data.data.length;
              if (jobCount > 0) {
                setResultData(prev => prev + `\n\n> 🎯 Found ${jobCount} job${jobCount !== 1 ? 's' : ''} for analysis.\n`);
              }
            } else if (data.type === 'result' && typeof data.message === 'string') {
              // Final consolidated result text from backend (e.g., full CV-job matching analysis)
              const text = data.message;
              setIsStreaming(false);
              setStreamingText('');
              setResultData(prev => (prev ? prev + "\n\n" : '') + text);
              setConversation(prev => [...prev, { role: 'assistant', content: text, timestamp: new Date().toISOString() }]);
              stopStream = true; // allow early termination after final result
            } else if (data.type === 'job' && data.data) {
              // Stream individual job items
              setJobs(prev => {
                const next = [...prev, data.data];
                return next;
              });
            } else if (data.type === 'markdown_page') {
              // Collect crawled page markdown preview
              setPagesMarkdown(prev => [...prev, {
                url: data.url || '',
                depth: data.depth,
                is_job_page: data.is_job_page,
                chars: data.chars,
                preview: data.preview || ''
              }]);
            } else if (data.type === 'error' && data.message) {
              // Stream the error into the main result area like assistant output and stop
              const cleanMessage = (data.message || 'An error occurred').toString();
              // Append to streaming text and result area immediately
              setIsStreaming(true);
              setStreamingText(prev => (prev ? prev + "\n" : "") + cleanMessage);
              setResultData(prev => (prev ? prev + "\n\n" : "") + cleanMessage);
              // Do not add to status history/log; keep it only as a response
              // End deep crawl indicator if any
              setDeepCrawlActive(false);
              // Signal to stop further streaming
              stopStream = true;
            } else if (data.type === 'intent' && data.data) {
              // Enhanced intent logging with workflow info
              console.log('🎯 Detected intent:', data.data);
              const workflow = data.data.workflow || 'general';
              const companies = data.data.company_names || [];
              const urls = data.data.urls || [];
              
              if (workflow !== 'general') {
                console.log(`📋 Workflow: ${workflow}`);
                if (companies.length > 0) console.log(`🏢 Companies: ${companies.join(', ')}`);
                if (urls.length > 0) console.log(`🔗 URLs: ${urls.join(', ')}`);
              }
            } else if (data.type === 'final') {
              // Two shapes possible: with final_response, or only meta (workflow/tools_used)
              const tryParseJobsArray = (text) => {
                try {
                  if (!text) return null;
                  const trimmed = text.trim();
                  if (!trimmed.startsWith('[')) return null;
                  const arr = JSON.parse(trimmed);
                  if (Array.isArray(arr) && arr.length >= 0) return arr;
                } catch {}
                return null;
              };

              if (typeof data.final_response === 'string') {
                const text = data.final_response || '';
                const arr = tryParseJobsArray(text);
                setIsStreaming(false);
                setStreamingText('');
                if (arr) {
                  setJobs(arr);
                  // Persist jobs by session id for later retrieval
                  try {
                    const sid = sessionId || data.session_id || '';
                    if (sid) {
                      localStorage.setItem(`tm_jobs_${sid}`, JSON.stringify(arr));
                    }
                  } catch {}
                  // Optional: show a compact summary instead of raw JSON
                  setResultData(prev => (prev ? prev + "\n\n" : '') + `Found ${arr.length} job${arr.length !== 1 ? 's' : ''}.`);
                  setConversation(prev => [...prev, { role: 'assistant', content: `Found ${arr.length} job${arr.length !== 1 ? 's' : ''}.`, timestamp: new Date().toISOString() }]);
                } else {
                  setResultData(prev => (prev ? prev + "\n\n" : '') + text);
                  setConversation(prev => [...prev, { role: 'assistant', content: text, timestamp: new Date().toISOString() }]);
                }
                if (data.workflow) setWorkflow(String(data.workflow));
              } else if (assistantAccum) {
                // Commit any accumulated streamed text as the assistant's final message
                setIsStreaming(false);
                setStreamingText('');
                const finalText = assistantAccum; // snapshot to avoid closure on mutable var
                const arr = tryParseJobsArray(finalText);
                if (arr) {
                  setJobs(arr);
                  // Persist jobs by session id for later retrieval
                  try {
                    const sid = sessionId || '';
                    if (sid) {
                      localStorage.setItem(`tm_jobs_${sid}`, JSON.stringify(arr));
                    }
                  } catch {}
                  setResultData(prev => (prev ? prev + "\n\n" : '') + `Found ${arr.length} job${arr.length !== 1 ? 's' : ''}.`);
                  setConversation(prev => [...prev, { role: 'assistant', content: `Found ${arr.length} job${arr.length !== 1 ? 's' : ''}.`, timestamp: new Date().toISOString() }]);
                } else {
                  setResultData(prev => (prev ? prev + "\n\n" : '') + finalText);
                  setConversation(prev => [...prev, { role: 'assistant', content: finalText, timestamp: new Date().toISOString() }]);
                }
              }
              if (process.env.NODE_ENV !== 'production') {
                try { console.log('[stream][final]'); } catch {}
              }
              // Reset active status on final
              setCurrentStatus('');
              setProgressPercent(100);
              setProgressStage('done');
              setProgressMessage('Completed');
              stopStream = true;
            } else if (data.type === 'plan') {
              // Capture current workflow early for UI decisions
              if (data.workflow) setWorkflow(String(data.workflow));
            } else if (data.type === 'thinking_step' && data.phase && data.content) {
              // Collect structured thinking steps with metadata for rich rendering
              const step = {
                phase: (data.phase || '').toString(),
                content: data.content || '',
                confidence: data.confidence,
                timestamp: data.timestamp || new Date().toISOString(),
                metadata: data.metadata || {}
              };
              setThinkingSteps(prev => [...prev, step]);
              // Keep a compact status log entry (without bracketed title)
              const title = step.phase.replace(/_/g, ' ');
              const msg = `${title}: ${step.content}`;
              setStatusHistory(prev => [...prev, {
                message: msg,
                timestamp: Date.now(),
                type: 'info'
              }]);
              // Do not append raw bracketed lines to streaming text anymore
            }
            
            startIndex = jsonEnd + 1;
            jsonStart = buffer.indexOf('{', startIndex);
          } catch (e) {
            // If parsing fails, wait for more data
            break;
          }
          if (stopStream) break;
        }
        
        // Remove processed data from buffer
        if (startIndex > 0) {
          buffer = buffer.substring(startIndex);
        }
        if (stopStream) break;
      }

      // After stream is complete, finalize the response
      setIsStreaming(false);
      if (!stopStream) {
        setResultData(result); // Set final result if no streamed error was appended
        if (assistantAccum) {
          const finalText = assistantAccum; // snapshot
          setConversation(prev => [...prev, { role: 'assistant', content: finalText, timestamp: new Date().toISOString() }]);
        }
      }
      // Ensure we clear inline status when stream ends
      setCurrentStatus('');
      setProgressMessage('');
      // Refresh sessions list preview
      try { await refreshSessions(); } catch {}
      
      const newPrompt = prompt || (file ? "CV Analysis" : "");
      if (newPrompt) {
        setPrevPrompts(prev => [...new Set([...prev, newPrompt])]);
      }

    } catch (error) {
      console.error('Error sending request:', error);
      setResultData(prev => prev + `\n\nError: ${error.message}`);
    } finally {
      setLoading(false);
      setIsUploading(false);
      setUploadStatus('');
      setCurrentStatus(''); // Clear current status when done
      setProgressPercent(0);
      setProgressStage('');
      setProgressMessage('');
      setToolSteps([]);
      if (!file) {
        setFile(null); // Only clear file if not uploading a new one
      }
      setDeepCrawlActive(false);
    }
  }, [sessionId, API_BASE_URL, refreshSessions, provider, isStreaming, progressPercent]);

  

  // Open a persisted session without reprocessing
  const openSession = useCallback(async (sid) => {
    if (!sid) return;
    if (!isAuthenticated()) {
      // Visitors cannot open stored history
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/chat/${sid}`, {
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
        },
      });
      if (!res.ok) throw new Error(`Failed to load session ${sid}`);
      const data = await res.json();
      const messages = data.messages || [];
      // Find last user and assistant messages for display
      const lastUser = [...messages].reverse().find(m => m.role === 'user');
      const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
      setSessionId(data.session_id || sid);
      setRecentPrompt(lastUser?.content || "");
      setResultData(lastAssistant?.content || "");
      setInput("");
      setFile(null);
      setConversation(messages);
      // Load persisted jobs for this session if available
      try {
        const key = `tm_jobs_${data.session_id || sid}`;
        const raw = localStorage.getItem(key);
        if (raw) {
          const arr = JSON.parse(raw);
          if (Array.isArray(arr)) setJobs(arr);
        } else {
          setJobs([]);
        }
      } catch { setJobs([]); }
    } catch (e) {
      setResultData(prev => prev + `\n\nError loading chat: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  const cleanup = () => {
    // Any cleanup needed when component unmounts
  };

  // Add cleanup on unmount
  useEffect(() => {
    // Load sessions on mount only if authenticated
    if (isAuthenticated()) {
      refreshSessions();
    } else {
      setSessions([]);
    }
    return () => cleanup();
  }, [refreshSessions]);

  return (
    <Context.Provider
      value={{
        // Form state
        input,
        setInput,
        recentPrompt,
        setRecentPrompt,
        prevPrompts,
        setPrevPrompts,
        showResult,
        setShowResult,
        loading,
        setLoading,
        resultData,
        setResultData,
        
        // File and session state
        file,
        setFile,
        sessionId,
        setSessionId,
        isUploading,
        setIsUploading,
        uploadStatus,
        setUploadStatus,
        currentStatus,
        setCurrentStatus,
        statusHistory,
        setStatusHistory,
        streamingText,
        setStreamingText,
        isStreaming,
        setIsStreaming,
        
        // Methods
        onSent,
        newChat,
        // Sessions API
        sessions,
        refreshSessions,
        openSession,
        jobs,
        setJobs,
        deepCrawlActive,
        setDeepCrawlActive,
        crawledBaseUrl,
        setCrawledBaseUrl,
        pagesMarkdown,
        setPagesMarkdown,
        thinkingSteps,
        setThinkingSteps,
        conversation,
        setConversation,
        // Provider selection
        provider,
        setProvider,
        // Workflow
        workflow,
        setWorkflow,
        // Structured progress
        progressPercent,
        setProgressPercent,
        progressStage,
        setProgressStage,
        progressMessage,
        setProgressMessage,
        toolSteps,
        setToolSteps,
      }}
    >
      {props.children}
    </Context.Provider>
  );
};

export { ContextProvider };
export default ContextProvider;