// components/Main/Main.jsx
import "./Main.css";
import { useContext, useEffect, useRef, useState } from "react";
import { Context } from "../../context/Context";
import { User, Brain, FileText, Send, Paperclip, Upload, Image, Mic, TrendingUp, CheckCircle, XCircle } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

const Main = () => {
  const location = useLocation();
  const isChat = (location?.pathname || '').startsWith('/chat');
  const {
    input,
    setInput,
    onSent,
    showResult,
    loading,
    file,
    sessionId,
    isUploading,
    uploadStatus,
    setFile,
    currentStatus,
    statusHistory,
    streamingText,
    isStreaming,
    conversation,
    provider,
    setProvider
  } = useContext(Context);
  const {
    progressPercent,
    progressStage,
    progressMessage,
    toolSteps,
    jobs,
    workflow,
  } = useContext(Context);

  const fileInputRef = useRef(null);
  const inputRef = useRef(null);
  const [fontScale, setFontScale] = useState(1);
  const [selectedQuestion, setSelectedQuestion] = useState(null);
  const [skillsModalOpen, setSkillsModalOpen] = useState(false);
  const [skillsForModal, setSkillsForModal] = useState([]);
  // removed unused skillsSourceIndex to satisfy linter
  const [jobsModalOpen, setJobsModalOpen] = useState(false);
  const [jobsForModal, setJobsForModal] = useState([]);

  // (removed unused tryParseJSON)
  
  // Render helper: color thinking steps in blue, AI text in default dark
  const renderAssistantText = (text) => {
    if (!text) return null;
    const phaseRegex = /^\[(initial analysis|planning|tool selection|execution|observation|reflection|final synthesis)\]/i;
    return text.split('\n').map((line, idx) => {
      const isThinking = phaseRegex.test(line.trim());
      return (
        <div key={idx} style={{ color: isThinking ? '#1976d2' : '#333' }}>
          {line}
        </div>
      );
    });
  };

  // Heuristic: detect whether a user's prompt is about job search/extraction
  const isJobIntent = (text) => {
    if (!text || typeof text !== 'string') return false;
    const t = text.toLowerCase();
    const patterns = [
      'job', 'jobs', 'opening', 'vacancy', 'vacancies', 'position', 'positions',
      'search', 'find', 'apply', 'hiring', 'career page', 'careers', 'listing', 'listings',
      'extract jobs', 'job extraction', 'compare my cv with', 'search for jobs', 'find roles'
    ];
    if (/https?:\/\//i.test(t)) return true; // URLs likely indicate job pages
    return patterns.some(p => t.includes(p));
  };

  // Try to parse job listings from assistant text. Prefer JSON array output.
  const extractJobsFromText = (text) => {
    if (!text || typeof text !== 'string') return [];
    // Attempt to find a JSON array in the text
    const jsonArrayMatch = text.match(/\[([\s\S]*?)\]/);
    if (jsonArrayMatch) {
      try {
        const arr = JSON.parse(jsonArrayMatch[0]);
        if (Array.isArray(arr)) {
          return arr
            .map(j => (typeof j === 'object' && j) ? j : null)
            .filter(Boolean)
            .map(j => ({
              title: j.title || j.job_title || j.position || '',
              company: j.company || j.employer || '',
              location: j.location || '',
              description: j.description || j.summary || '',
              url: j.url || j.apply_url || j.link || '',
              salary: j.salary || j.compensation || '',
              contractType: j.contractType || j.contract_type || j['contrat-type'] || '',
              requirements: j.requirements || j.requirement || '',
              requiredSkill: j.requiredSkill || j.required_skill || '',
              postDate: j.postDate || j.post_date || j.date || '',
            }))
            .filter(j => j.title || j.company || j.description || j.url);
        }
      } catch (e) {
        // ignore JSON parse error; fall through to heuristic
      }
    }
    // Attempt to parse a single JSON object if present
    const jsonObjectMatch = text.trim().match(/^\{[\s\S]*\}$/);
    if (jsonObjectMatch) {
      try {
        const j = JSON.parse(jsonObjectMatch[0]);
        if (j && typeof j === 'object') {
          const job = {
            title: j.title || j.job_title || j.position || '',
            company: j.company || j.employer || '',
            location: j.location || '',
            description: j.description || j.summary || '',
            url: j.url || j.apply_url || j.link || '',
            salary: j.salary || j.compensation || '',
            contractType: j.contractType || j.contract_type || j['contrat-type'] || '',
            requirements: j.requirements || j.requirement || '',
            requiredSkill: j.requiredSkill || j.required_skill || '',
            postDate: j.postDate || j.post_date || j.date || '',
          };
          if (job.title || job.company || job.description || job.url) return [job];
        }
      } catch (e) {
        // ignore
      }
    }
    // Lightweight heuristic: look for lines like "Title - Company - Location"
    const lines = text.split(/\r?\n/);
    const jobs = [];
    for (const raw of lines) {
      const line = (raw || '').trim();
      if (!line) continue;
      // basic split by ' - '
      const parts = line.split(' - ').map(s => s.trim());
      if (parts.length >= 2 && parts[0].length > 2) {
        jobs.push({
          title: parts[0],
          company: parts[1] || '',
          location: parts[2] || '',
          description: '',
          url: ''
        });
      }
    }
    return jobs;
  };

  // (removed unused prettyPhase and renderStepDetails)

  // Extract skills from a block of text. Looks for "Skills" headings or lines like "Skills: ..."
  const extractSkillsFromText = (text) => {
    if (!text || typeof text !== 'string') return [];
    const lines = text.split(/\r?\n/);
    let skillsSection = [];
    let capturing = false;

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i] || '';
      const line = raw.trim();
      const lower = line.toLowerCase();

      // Explicit inline list: "Skills: a, b, c"
      if (lower.startsWith('skills:') || lower.startsWith('key skills:')) {
        const after = line.split(':').slice(1).join(':');
        const parts = after.split(/[,•\u2022]/).map(s => s.trim()).filter(Boolean);
        return Array.from(new Set(parts.map(p => p.replace(/^[-*•\u2022]+\s*/, '').trim())));
      }

      // Start capturing after a "Skills" heading
      if (!capturing && (/^\s*#+\s*(key\s+)?skills\b/i.test(line) || /^(key\s+)?skills\b[:]?$/i.test(line))) {
        capturing = true;
        continue;
      }
      if (capturing) {
        if (line === '' || /^\s*#+\s+/.test(line)) break; // stop at blank line or next heading
        skillsSection.push(line);
      }
    }

    // Parse bullets and comma-separated lists from captured section
    const items = [];
    skillsSection.forEach(l => {
      const cleaned = l.replace(/^[-*•\u2022]+\s*/, '').trim();
      if (!cleaned) return;
      if (/[,;]\s/.test(cleaned)) {
        cleaned.split(/[;,]/).forEach(p => { const t = p.trim(); if (t) items.push(t); });
      } else {
        items.push(cleaned);
      }
    });

    // Fallback: search anywhere for a one-line skills list
    if (items.length === 0) {
      const m = text.match(/\b(key\s+)?skills\b\s*[:|-]\s*([^\n]+)/i);
      if (m && m[2]) {
        m[2].split(/[;,]/).forEach(p => { const t = p.trim(); if (t) items.push(t); });
      }
    }

    // Normalize and uniq
    return Array.from(new Set(items.map(s => s.replace(/^[-*•\u2022]+\s*/, '').trim()).filter(Boolean)));
  };

  // Adjust textarea height and font size based on content
  const adjustInputMetrics = () => {
    const el = inputRef.current;
    if (!el) return;
    // Auto-resize height up to a max
    const maxHeight = 120; // px
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, maxHeight) + 'px';

    // Compute overflow ratio (horizontal) and length-based fallback
    const width = el.clientWidth || 1;
    const scrollW = el.scrollWidth || width;
    const ratio = Math.max(1, scrollW / width);
    const byOverflow = 1 / ratio; // 1 means fits, <1 means shrink
    const len = (el.value || '').length;
    const byLength = len <= 60 ? 1 : Math.max(0.75, 60 / len);
    // Pick the most conservative (smallest) scale within [0.75, 1]
    const target = Math.max(0.75, Math.min(1, Math.min(byOverflow, byLength)));
    // Smooth the changes a bit
    setFontScale(prev => prev * 0.8 + target * 0.2);
  };

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile); // Just attach the file, don't auto-send
      } else {
        alert('Please select a PDF file');
      }
      // Reset file input so the same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleCardClick = (prompt) => {
    setSelectedQuestion(prompt);
    setInput(prompt);
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      const toSend = selectedQuestion || input;
      if (toSend || file) {
        onSent(toSend, file);
        // If we used a selected question, clear selection after sending
        if (selectedQuestion) setSelectedQuestion(null);
      }
    }
  };

  useEffect(() => {
    adjustInputMetrics();
    // Re-adjust on window resize
    const onResize = () => adjustInputMetrics();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // (removed unused slow-step tracking)

  useEffect(() => {
    adjustInputMetrics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input]);

  // Auto-open Jobs modal when structured jobs arrive from backend context
  useEffect(() => {
    try {
      if (workflow !== 'cv_job_matching' && Array.isArray(jobs) && jobs.length > 0) {
        const normalized = jobs
          .map(j => j && typeof j === 'object' ? j : null)
          .filter(Boolean)
          .map(j => ({
            title: j.title || j.job_title || j.position || 'Untitled role',
            company: j.company || j.employer || '',
            location: j.location || '',
            description: j.description || j.summary || '',
            url: j.url || j.apply_url || j.link || '',
            salary: j.salary || j.compensation || '',
            contractType: j.contractType || j.contract_type || j['contrat-type'] || '',
            requirements: j.requirements || j.requirement || '',
            requiredSkill: j.requiredSkill || j.required_skill || '',
            postDate: j.postDate || j.post_date || j.date || '',
          }));
        setJobsForModal(normalized);
        setJobsModalOpen(true);
      }
    } catch {}
    // React to jobs and workflow changes
  }, [jobs, workflow]);

  const cvQuestions = [
    "Can you summarize this CV for me?",
    "What are the key skills and strengths mentioned?", 
    "What improvements would you suggest for this CV?",
    "Analyze the work experience and career progression"
  ];

  return (
    <>
      <div className="main">
        <div className="nav">
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <User size={40} style={{ backgroundColor: '#4B90FF', color: 'white', borderRadius: '50%', padding: '8px' }} />
            {sessionId && file && (
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px', 
                padding: '8px 12px', 
                backgroundColor: '#E6F7FF', 
                borderRadius: '20px',
                fontSize: '14px',
                color: '#1890ff'
              }}>
                <FileText size={16} />
                <span>{file.name}</span>
              </div>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <p style={{ margin: 0 }}>TalentMind</p>
            {/* Quick access Jobs button if any jobs exist in context */}
            {workflow !== 'cv_job_matching' && Array.isArray(jobs) && jobs.length > 0 && (
              <button
                className="jobs-toggle-btn"
                onClick={() => {
                  try {
                    const normalized = jobs
                      .map(j => j && typeof j === 'object' ? j : null)
                      .filter(Boolean)
                      .map(j => ({
                        title: j.title || j.job_title || j.position || 'Untitled role',
                        company: j.company || j.employer || '',
                        location: j.location || '',
                        description: j.description || j.summary || '',
                        url: j.url || j.apply_url || j.link || '',
                        salary: j.salary || j.compensation || '',
                        contractType: j.contractType || j.contract_type || j['contrat-type'] || '',
                        requirements: j.requirements || j.requirement || '',
                        requiredSkill: j.requiredSkill || j.required_skill || '',
                        postDate: j.postDate || j.post_date || j.date || '',
                      }));
                    setJobsForModal(normalized);
                    setJobsModalOpen(true);
                  } catch {}
                }}
                title="View extracted jobs"
                style={{ marginLeft: 8 }}
              >
                Jobs ({jobs.length})
              </button>
            )}
          </div>
        </div>
        
        <div className="main-container">
          {/* Global Error Banner (always visible when last status is error) */}
          {statusHistory.length > 0 && statusHistory[statusHistory.length - 1]?.type === 'error' && (
            <div className="error-banner" role="alert">
              <span style={{ fontWeight: 600, marginRight: 8 }}>Error:</span>
              <span>{statusHistory[statusHistory.length - 1]?.message || 'An error occurred.'}</span>
            </div>
          )}

          {/* Global Status Banner on home (non-result) to show progress */}
          {!showResult && currentStatus && (
            <div className="status-banner">
              <div className="status-dot" />
              <span>{currentStatus}</span>
            </div>
          )}
          {!(conversation && conversation.length > 0) ? (
            <>
              <div className="greet">
                <p>
                  <span>{(() => {
                    try {
                      const token = localStorage.getItem('tm_token');
                      const raw = localStorage.getItem('tm_user');
                      const parsed = raw ? JSON.parse(raw) : null;
                      const displayName = ((parsed?.name || parsed?.username || parsed?.email || '') + '').trim();
                      if (isChat && token) return `Welcome, ${displayName || 'Professional'}`;
                      return 'Welcome to TalentMind';
                    } catch {
                      return 'Welcome to TalentMind';
                    }
                  })()}</span>
                </p>
                <p> How can I help you unlock your career potential today ? </p>
                {!isChat && (() => {
                  try {
                    const token = localStorage.getItem('tm_token');
                    if (!token) {
                      return (
                        <div style={{ marginTop: '12px' }}>
                          <Link to="/signin" className="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors">
                            Sign In
                          </Link>
                        </div>
                      );
                    }
                    return null;
                  } catch {
                    return (
                      <div style={{ marginTop: '12px' }}>
                        <Link to="/signin" className="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors">
                          Sign In
                        </Link>
                      </div>
                    );
                  }
                })()}
              </div>

              {/* Upload Status */}
              {isUploading && (
                <div className="upload-status">
                  <FileText size={20} style={{ marginRight: '8px' }} />
                  <span>Uploading {file?.name}...</span>
                </div>
              )}
              
              {/* Upload Status Error */}
              {uploadStatus && uploadStatus.startsWith('❌') && (
                <div style={{ 
                  margin: '20px', 
                  padding: '15px', 
                  borderRadius: '10px', 
                  textAlign: 'center',
                  backgroundColor: uploadStatus.includes('❌') ? '#ffebee' : '#e8f5e8',
                  color: uploadStatus.includes('❌') ? '#c62828' : '#2e7d32',
                  border: uploadStatus.includes('❌') ? '1px solid #ffcdd2' : '1px solid #c8e6c9'
                }}>
                  {uploadStatus}
                </div>
              )}

              {/* Loading Animation */}
              {isUploading && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '40px 0' }}>
                  <p style={{ marginTop: '0', color: '#585858' }}>Processing your CV...</p>
                </div>
              )}

              {/* CV Analysis Cards - show when a session exists or a CV file is attached */}
              {!isUploading && (sessionId || file) && (
                <>
                  {/* Ask about your CV heading removed */}
                  <div className="cards">
                    {cvQuestions.map((question, index) => (
                      <div 
                        key={index} 
                        className="card"
                        onClick={() => handleCardClick(question)}
                      >
                        <p style={{
                          fontWeight: selectedQuestion === question ? 600 : 400,
                          color: selectedQuestion === question ? '#4B90FF' : 'inherit'
                        }}>{question}</p>
                      </div>
                    ))}
                  </div>
                  {/* Removed Confirm & Send / Clear buttons under CV cards */}
                  {(!sessionId && !file) && (
                    <>
                      <h3 style={{ marginTop: '2rem' }}>Job Search & Comparison</h3>
                      <div className="cards">
                        <div 
                          className="card"
                          onClick={() => handleCardClick("Find jobs that match my CV at Google")}
                        >
                          <p>Find jobs that match my CV at Google</p>
                        </div>
                        <div 
                          className="card"
                          onClick={() => handleCardClick("Compare my CV with Microsoft jobs")}
                        >
                          <p>Compare my CV with Microsoft jobs</p>
                        </div>
                        <div 
                          className="card"
                          onClick={() => handleCardClick("Search for jobs at https://www.odoo.com/jobs")}
                        >
                          <p>Search jobs from URL</p>
                        </div>
                        <div 
                          className="card"
                          onClick={() => handleCardClick("How does my CV look for Amazon roles?")}
                        >
                          <p>How does my CV look for Amazon roles?</p>
                        </div>
                      </div>
                    </>
                  )}
                </>
              )}


              {/* Suggestion Cards - No CV uploaded (only when no session, no file, and no result shown) */}
              {!sessionId && !isUploading && !file && !showResult && (
                <div className="cards">
                  <div className="card" onClick={() => handleCardClick("Can you summarize this CV for me?")}>
                    <p style={{ fontWeight: selectedQuestion === "Can you summarize this CV for me?" ? 600 : 400, color: selectedQuestion === "Can you summarize this CV for me?" ? '#4B90FF' : 'inherit' }}>Can you summarize this CV for me?</p>
                    <Upload size={35} className="card-icon" />
                  </div>
                  <div className="card" onClick={() => handleCardClick("What are the key skills and strengths mentioned?")}>
                    <p style={{ fontWeight: selectedQuestion === "What are the key skills and strengths mentioned?" ? 600 : 400, color: selectedQuestion === "What are the key skills and strengths mentioned?" ? '#4B90FF' : 'inherit' }}>What are the key skills and strengths mentioned?</p>
                    <TrendingUp size={35} className="card-icon" />
                  </div>
                  <div className="card" onClick={() => handleCardClick("What improvements would you suggest for this CV?")}>
                    <p style={{ fontWeight: selectedQuestion === "What improvements would you suggest for this CV?" ? 600 : 400, color: selectedQuestion === "What improvements would you suggest for this CV?" ? '#4B90FF' : 'inherit' }}>What improvements would you suggest for this CV?</p>
                    <FileText size={35} className="card-icon" />
                  </div>
                  <div className="card" onClick={() => handleCardClick("Analyze the work experience and career progression")}>
                    <p style={{ fontWeight: selectedQuestion === "Analyze the work experience and career progression" ? 600 : 400, color: selectedQuestion === "Analyze the work experience and career progression" ? '#4B90FF' : 'inherit' }}>Analyze the work experience and career progression</p>
                    <Brain size={35} className="card-icon" />
                  </div>
                  {/* Removed Confirm & Send / Clear buttons under suggestion cards */}
                </div>
              )}
            </>
          ) : (
            <div className="result">

              {/* Conversation thread */}
              <div className="result-data" style={{ display: 'grid', gap: 16 }}>
                {/* Removed global spinner; progress now shows inline under latest user message */}
                {(() => {
                  const lastUserIdx = (() => {
                    let idx = -1;
                    for (let i = conversation.length - 1; i >= 0; i--) {
                      if (conversation[i]?.role === 'user') { idx = i; break; }
                    }
                    return idx;
                  })();
                  return conversation.map((m, idx) => (
                    <div key={idx} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                      {m.role === 'assistant' ? (
                        <Brain size={32} style={{ color: '#4B90FF', marginTop: 2 }} />
                      ) : (
                        <User size={32} style={{ marginTop: 2 }} />
                      )}
                      <div style={{ whiteSpace: 'pre-wrap', flex: 1 }}>
                        {m.role === 'assistant' ? (
                          <div>{renderAssistantText(m.content)}</div>
                        ) : (
                          <div style={{ color: '#263238' }}>{m.content}</div>
                        )}
                        {/* Skills quick view button under assistant messages if skills detected */}
                        {m.role === 'assistant' && (() => {
                          const skills = extractSkillsFromText(m.content);
                          if (skills && skills.length > 0) {
                            return (
                              <div style={{ marginTop: 8 }}>
                                <button
                                  className="jobs-toggle-btn"
                                  onClick={() => { setSkillsForModal(skills); setSkillsModalOpen(true); }}
                                  title="View skills extracted from this answer"
                                >
                                  View Skills ({skills.length})
                                </button>
                              </div>
                            );
                          }
                          return null;
                        })()}
                        {/* Jobs quick view button only for job-search intents */}
                        {m.role === 'assistant' && (() => {
                          // Find the most recent user message before this assistant message
                          let prevUser = null;
                          for (let i = Math.min(idx - 1, conversation.length - 1); i >= 0; i--) {
                            if (conversation[i]?.role === 'user') { prevUser = conversation[i]; break; }
                          }
                          const jobs = extractJobsFromText(m.content);
                          if (jobs && jobs.length > 0 && prevUser && isJobIntent(prevUser.content)) {
                            return (
                              <div style={{ marginTop: 8 }}>
                                <button
                                  className="jobs-toggle-btn"
                                  onClick={() => { setJobsForModal(jobs); setJobsModalOpen(true); }}
                                  title="View extracted jobs"
                                >
                                  View Jobs ({jobs.length})
                                </button>
                              </div>
                            );
                          }
                          return null;
                        })()}
                        {/* Inline loading status under the latest user message */}
                        {m.role === 'user' && idx === lastUserIdx && (isStreaming || (progressPercent > 0) || (currentStatus && currentStatus.length > 0)) && (
                          <div style={{
                            marginTop: 10,
                            display: 'grid',
                            gap: 8,
                            padding: '10px 12px',
                            backgroundColor: '#f8fbff',
                            border: '1px solid #e6f2ff',
                            borderRadius: 10,
                            color: '#0f3b6e'
                          }}>
                            {/* Progress header */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <div className="loading-spinner" style={{
                                width: 14,
                                height: 14,
                                border: '2px solid #e3f2fd',
                                borderTop: '2px solid #1976d2',
                                borderRadius: '50%'
                              }} />
                              <strong style={{ fontSize: 13 }}>
                                {progressStage ? progressStage.replace(/_/g, ' ') : 'working'}
                              </strong>
                              <span style={{ fontSize: 13, color: '#22577a' }}>{progressMessage || currentStatus}</span>
                              <span style={{ marginLeft: 'auto', fontSize: 12, color: '#336699' }}>{Math.max(0, Math.min(100, progressPercent || 0))}%</span>
                            </div>
                            {/* Progress bar */}
                            <div style={{ height: 8, background: '#e6f2ff', borderRadius: 6, overflow: 'hidden' }}>
                              <div style={{ width: `${Math.max(0, Math.min(100, progressPercent || 0))}%`, height: '100%', background: 'linear-gradient(90deg, #66a6ff, #4B90FF)', transition: 'width 200ms ease' }} />
                            </div>
                            {/* Tool steps checkmarks (only if any reported) */}
                            {toolSteps && toolSteps.length > 0 && (
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                                {toolSteps.map((t, i) => (
                                  <div key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 8px', borderRadius: 9999, background: '#eef6ff', border: '1px solid #dbeafe', fontSize: 12, color: t.success ? '#0f5132' : '#7f1d1d' }}>
                                    {t.success ? (
                                      <CheckCircle size={14} color="#16a34a" />
                                    ) : (
                                      <XCircle size={14} color="#dc2626" />
                                    )}
                                    <span style={{ color: '#0f3b6e' }}>{t.tool}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  ));
                })()}
                {isStreaming && streamingText && (
                  <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                    <Brain size={32} style={{ color: '#4B90FF', marginTop: 2 }} />
                    <div style={{ whiteSpace: 'pre-wrap', flex: 1 }}>
                      {renderAssistantText(streamingText)}
                      <span className="typing-cursor">|</span>
                    </div>
                  </div>
                )}

                {/* Reasoning panel removed per request */}
              </div>
            </div>
          )}
          {/* Skills Modal */}
          {skillsModalOpen && (
            <div className="jobs-modal-backdrop" onClick={() => setSkillsModalOpen(false)}>
              <div className="jobs-modal" onClick={(e) => e.stopPropagation()}>
                <div className="jobs-modal-header">
                  <strong>Skills</strong>
                  <button className="jobs-close-btn" onClick={() => setSkillsModalOpen(false)}>✕</button>
                </div>
                <div className="jobs-modal-body">
                  {skillsForModal && skillsForModal.length > 0 ? (
                    <ul style={{ paddingInlineStart: 20, margin: 0 }}>
                      {skillsForModal.map((s, i) => (
                        <li key={i} style={{ margin: '6px 0' }}>{s}</li>
                      ))}
                    </ul>
                  ) : (
                    <div>No skills detected.</div>
                  )}
                </div>
              </div>
            </div>
          )}
          {/* Jobs Modal */}
          {jobsModalOpen && (
            <div className="jobs-modal-backdrop" onClick={() => setJobsModalOpen(false)}>
              <div className="jobs-modal" onClick={(e) => e.stopPropagation()}>
                <div className="jobs-modal-header">
                  <strong>Extracted Jobs</strong>
                  <button className="jobs-close-btn" onClick={() => setJobsModalOpen(false)}>✕</button>
                </div>
                <div className="jobs-modal-body">
                  {jobsForModal && jobsForModal.length > 0 ? (
                    <div className="jobs-grid">
                      {jobsForModal.map((job, i) => (
                        <div key={i} className="job-card">
                          <div className="job-header">
                            <div className="job-title">{job.title || 'Untitled role'}</div>
                          </div>
                          <div className="job-meta">
                            {job.company && <span className="chip chip-company">{job.company}</span>}
                            {job.location && <span className="chip chip-location">{job.location}</span>}
                          </div>
                          {job.description && (
                            <div className="job-desc" style={{ whiteSpace: 'pre-wrap' }}>{job.description}</div>
                          )}
                          {/* Requirements block */}
                          {job.requirements && (
                            <div className="job-desc" style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>
                              <strong>Requirements</strong>
                              <div style={{ marginTop: 4 }}>{job.requirements}</div>
                            </div>
                          )}
                          {/* Extra meta chips */}
                          <div className="job-meta">
                            {job.salary && <span className="chip">{job.salary}</span>}
                            {job.contractType && <span className="chip">{job.contractType}</span>}
                            {job.requiredSkill && <span className="chip">{job.requiredSkill}</span>}
                            {job.postDate && <span className="chip">{job.postDate}</span>}
                          </div>
                          <div className="job-actions">
                            <a
                              className="apply-btn"
                              href={job.url || '#'}
                              target={job.url ? "_blank" : undefined}
                              rel={job.url ? "noreferrer" : undefined}
                              onClick={(e) => { if (!job.url) e.preventDefault(); }}
                            >
                              {job.url ? 'Apply' : 'No link'}
                            </a>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div>No jobs detected.</div>
                  )}
                </div>
              </div>
            </div>
          )}
          {/* Bottom Input Bar - Always visible */}
          <div className="main-bottom">
            {file && (
              <div className="file-attachment-indicator">
                <FileText size={18} />
                <span>{file.name}</span>
              </div>
            )}
            <div className="search-box" style={{ '--input-font-scale': fontScale }}>
              <textarea
                ref={inputRef}
                className="search-textarea"
                onChange={(event) => { setInput(event.target.value); /* adjust after state updates */ setTimeout(adjustInputMetrics, 0); }}
                value={input}
                placeholder={"Ask me anything about your career..."}
                onKeyPress={handleKeyPress}
                disabled={loading}
                rows={1}
              />
              <div className="search-box-icon">
                {/* CV Upload Button */}
                <div style={{ position: 'relative' }}>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    id="file-upload-input"
                  />
                  <label htmlFor="file-upload-input" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                     <Paperclip size={25} style={{ color: isUploading ? '#ccc' : '#4B90FF' }} />
                  </label>
                </div>
                {/* Provider compact selector inside input bar */}
                <select
                  aria-label="Model Provider"
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  style={{
                    appearance: 'none',
                    WebkitAppearance: 'none',
                    MozAppearance: 'none',
                    padding: '6px 8px',
                    borderRadius: 8,
                    border: '1px solid #dbeafe',
                    background: '#ffffff',
                    color: '#1f2937',
                    fontSize: 12,
                    cursor: 'pointer'
                  }}
                  title="Choose provider"
                >
                  <option value="local">Ollama</option>
                  <option value="groq">Groq</option>
                </select>
                <Image size={25} style={{ cursor: 'pointer', color: '#666' }} />
                <Mic size={25} style={{ cursor: 'pointer', color: '#666' }} />
                
                {(input || file || selectedQuestion) ? (
                  <Send
                    onClick={() => {
                      const toSend = selectedQuestion || input;
                      if (toSend || file) {
                        onSent(toSend, file);
                        if (selectedQuestion) setSelectedQuestion(null);
                      }
                    }}
                    size={25}
                    style={{ cursor: 'pointer', color: '#4B90FF' }}
                    title="Send (Enter) | New line (Shift+Enter)"
                  />
                ) : null}
              </div>
            </div>
            <p className="bottom-info">
              TalentMind may display inaccurate info about your CV, so double-check its responses.{" "}
              <span style={{ color: '#4B90FF', cursor: 'pointer' }}>
                Your privacy & TalentMind Apps
              </span>
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default Main;