import React, { useState, useRef, useEffect } from 'react';
import {
  Search, User, MapPin, DollarSign, Briefcase, Mail,
  Phone, ExternalLink, Loader2, Settings, X, ChevronDown,
  ChevronUp, Sparkles, Menu, File, Clock, Zap
} from 'lucide-react';

const Dashboard = () => {
  // ---------------- State ----------------
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [includeAnswer, setIncludeAnswer] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedCards, setExpandedCards] = useState({});
  const [countries, setCountries] = useState([]);
  const [domains, setDomains] = useState([]);
  const [requestedFields, setRequestedFields] = useState([]);
  const [limit, setLimit] = useState('');
  const [searchHistory, setSearchHistory] = useState([]);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const resultsRef = useRef(null);
  const textareaRef = useRef(null);

  // Normalize API base (strip trailing slashes)
  const RAW_API_BASE = 'http://localhost:8000/';
  const API_BASE_URL = RAW_API_BASE.replace(/\/+$/, '');

  // ---------------- Config ----------------
  const availableCountries = ['India', 'Japan', 'Bangladesh', 'Indonesia', 'Philippines', 'Pakistan', 'Morocco', 'Italy', 'Portugal'];
  const availableDomains = ['audio', 'web', 'pcb', 'devops', 'data_annot'];
  const availableFields = ['phone', 'email', 'profile', 'hourly_rate_num', 'location_region', 'skills'];
  
  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [query]);

  // ---------------- Handlers ----------------
  const simulateStreaming = (text, callback) => {
    let index = 0;
    const words = text.split(' ');
    
    const interval = setInterval(() => {
      if (index < words.length) {
        callback(words.slice(0, index + 1).join(' '));
        index++;
      } else {
        clearInterval(interval);
        setIsGenerating(false);
      }
    }, 50);
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query.');
      return;
    }

    setLoading(true);
    setIsGenerating(true);
    setError('');
    setStreamingAnswer('');
    setResults(null);
    setExpandedCards({});

    const payload = {
      query: query.trim(),
      include_answer: includeAnswer,
      answer_model: "llama-3.3-70b-versatile",
      ...(countries.length > 0 && { countries }),
      ...(domains.length > 0 && { domains }),
      ...(requestedFields.length > 0 && { requested_fields: requestedFields }),
      ...(limit && { limit: parseInt(limit, 10) || undefined })
    };

    // Add to history
    setSearchHistory(prev => [{
      query: query.trim(),
      timestamp: new Date().toISOString()
    }, ...prev.slice(0, 9)]);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const text = await response.text();

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} - ${text}`);
      }

      let data;
      try {
        data = JSON.parse(text);
      } catch (jsonErr) {
        throw new Error(`Invalid JSON response: ${jsonErr.message}`);
      }

      // Simulate streaming for the answer
      if (data.answer && includeAnswer) {
        simulateStreaming(data.answer, setStreamingAnswer);
      } else {
        setIsGenerating(false);
      }

      setResults(data);
      
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      console.error('[ERROR] Search failed', err);
      setError(err.message || 'Failed to fetch results. Ensure backend is running on port 8000.');
      setIsGenerating(false);
    } finally {
      setLoading(false);
    }
  };

  const toggleArrayFilter = (arr, setArr, value) => {
    setArr(arr.includes(value) ? arr.filter(v => v !== value) : [...arr, value]);
  };

  const clearFilters = () => {
    setCountries([]);
    setDomains([]);
    setRequestedFields([]);
    setLimit('');
  };

  const toggleCardExpansion = (id) => {
    setExpandedCards(prev => ({ ...prev, [id]: !prev[id] }));
  };

  // ---------------- Render ----------------
  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 border-r border-gray-200 bg-gray-50 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2 mb-6">
            <Sparkles className="w-6 h-6 text-blue-600" />
            <h1 className="text-lg font-semibold text-gray-900">Applicant Search</h1>
          </div>
          
          <button
            onClick={() => {
              setQuery('');
              setResults(null);
              setError('');
              setStreamingAnswer('');
            }}
            className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            <Sparkles className="w-4 h-4" />
            New Search
          </button>
        </div>

        {/* Search History */}
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-3">Recent Searches</h3>
          <div className="space-y-2">
            {searchHistory.map((item, idx) => (
              <button
                key={idx}
                onClick={() => setQuery(item.query)}
                className="w-full text-left p-3 rounded-lg hover:bg-gray-100 transition-colors group"
              >
                <div className="flex items-start gap-2">
                  <Clock className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-700 truncate group-hover:text-gray-900">
                      {item.query}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {new Date(item.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </button>
            ))}
            {searchHistory.length === 0 && (
              <p className="text-sm text-gray-400 text-center py-8">No recent searches</p>
            )}
          </div>
        </div>

        {/* Settings Section */}
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={() => setSidebarOpen(false)}
            className="w-full px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors flex items-center gap-2"
          >
            <Settings className="w-4 h-4" />
            Settings
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="border-b border-gray-200 px-6 py-3 flex items-center gap-4 bg-white">
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Menu className="w-5 h-5 text-gray-600" />
            </button>
          )}
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <File className="w-4 h-4" />
            <span>Semantic Search RAG System</span>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-6 py-8">
            
            {/* Welcome State */}
            {!results && !loading && (
              <div className="text-center py-16">
                <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <Sparkles className="w-8 h-8 text-blue-600" />
                </div>
                <h2 className="text-3xl font-semibold text-gray-900 mb-3">
                  Find Technical Talent
                </h2>
                <p className="text-gray-600 mb-8 max-w-lg mx-auto">
                  Search through applicants using natural language. Our AI will understand your requirements and find the best matches.
                </p>
              </div>
            )}

            {/* Results Area */}
            {(results || loading) && (
              <div ref={resultsRef} className="space-y-6 mb-8">
                {/* AI Summary */}
                {(streamingAnswer || (results?.answer && includeAnswer)) && (
                  <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                        <Sparkles className="w-4 h-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-gray-900">AI Summary</h3>
                      {isGenerating && (
                        <Loader2 className="w-4 h-4 text-blue-600 animate-spin ml-auto" />
                      )}
                    </div>
                    <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                      {streamingAnswer || results.answer}
                      {isGenerating && <span className="inline-block w-2 h-5 bg-blue-600 ml-1 animate-pulse"></span>}
                    </div>
                  </div>
                )}

                {/* Results Count */}
                {results && (
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-600">
                      Found <span className="font-semibold text-gray-900">{results.results.length}</span> matching applicants
                    </p>
                    {results.parsed_intent && (
                      <div className="flex items-center gap-2">
                        {results.parsed_intent.countries?.length > 0 && (
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                            📍 {results.parsed_intent.countries.join(', ')}
                          </span>
                        )}
                        {results.parsed_intent.domains?.length > 0 && (
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                            💼 {results.parsed_intent.domains.join(', ')}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Applicant Cards */}
                {results?.results && results.results.length > 0 && (
                  <div className="space-y-4">
                    {results.results.map(applicant => {
                      const isExpanded = expandedCards[applicant.applicant_id];
                      return (
                        <div
                          key={applicant.applicant_id}
                          className="bg-white rounded-xl p-6 border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all"
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex items-start gap-4 flex-1">
                              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
                                <User className="w-6 h-6 text-white" />
                              </div>
                              <div className="flex-1">
                                <h3 className="text-lg font-semibold text-gray-900">
                                  {applicant.name || `Applicant #${applicant.applicant_id}`}
                                </h3>
                                <p className="text-gray-600 text-sm">{applicant.title || '—'}</p>
                                
                                <div className="flex flex-wrap gap-3 mt-3 text-sm text-gray-600">
                                  {applicant.location_region && (
                                    <div className="flex items-center gap-1.5">
                                      <MapPin className="w-4 h-4" />
                                      <span>{applicant.country_code}, {applicant.location_region}</span>
                                    </div>
                                  )}
                                  {applicant.hourly_rate_num && (
                                    <div className="flex items-center gap-1.5">
                                      <DollarSign className="w-4 h-4" />
                                      <span>${applicant.hourly_rate_num}/hr</span>
                                    </div>
                                  )}
                                  {applicant.experience_years_est && (
                                    <div className="flex items-center gap-1.5">
                                      <Briefcase className="w-4 h-4" />
                                      <span>{applicant.experience_years_est} years</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                            <div className="px-3 py-1 bg-blue-50 rounded-lg">
                              <span className="text-blue-700 text-sm font-medium">
                                {applicant.score.toFixed(2)}
                              </span>
                            </div>
                          </div>

                          {/* Contact Info */}
                          {(applicant.email || applicant.phone || applicant.profile_url) && (
                            <div className="flex flex-wrap gap-4 mb-4 pb-4 border-b border-gray-100">
                              {applicant.email && (
                                <a href={`mailto:${applicant.email}`} className="flex items-center gap-2 text-sm text-gray-600 hover:text-blue-600 transition-colors">
                                  <Mail className="w-4 h-4" />
                                  <span>{applicant.email}</span>
                                </a>
                              )}
                              {applicant.phone && (
                                <a href={`tel:${applicant.phone}`} className="flex items-center gap-2 text-sm text-gray-600 hover:text-blue-600 transition-colors">
                                  <Phone className="w-4 h-4" />
                                  <span>{applicant.phone}</span>
                                </a>
                              )}
                              {applicant.profile_url && (
                                <a
                                  href={applicant.profile_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors"
                                >
                                  <ExternalLink className="w-4 h-4" />
                                  <span>View Profile</span>
                                </a>
                              )}
                            </div>
                          )}

                          {/* Skills */}
                          {applicant.top_skills?.length > 0 && (
                            <div className="mb-4">
                              <h4 className="text-sm font-medium text-gray-700 mb-2">Top Skills</h4>
                              <div className="flex flex-wrap gap-2">
                                {applicant.top_skills
                                  .slice(0, isExpanded ? undefined : 8)
                                  .map((skill, idx) => (
                                    <span
                                      key={idx}
                                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-xs"
                                    >
                                      {skill}
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}

                          {/* Snippets */}
                          {applicant.snippets?.length > 0 && (
                            <div>
                              <button
                                onClick={() => toggleCardExpansion(applicant.applicant_id)}
                                className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                              >
                                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                {isExpanded ? 'Hide' : 'Show'} Profile Details ({applicant.snippets.length})
                              </button>
                              {isExpanded && (
                                <div className="mt-3 space-y-2 pl-6 border-l-2 border-gray-200">
                                  {applicant.snippets.map((snippet, idx) => (
                                    <p key={idx} className="text-sm text-gray-600 leading-relaxed">
                                      {snippet}
                                    </p>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}

                {results?.results?.length === 0 && (
                  <div className="text-center py-12">
                    <p className="text-gray-600">No applicants found matching your criteria.</p>
                    <p className="text-sm text-gray-400 mt-2">Try adjusting your search query or filters.</p>
                  </div>
                )}
              </div>
            )}

            {/* Error Banner */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
                <p className="text-red-800 text-sm">⚠️ {error}</p>
              </div>
            )}
          </div>
        </div>

        {/* Input Area - Fixed at Bottom */}
        <div className="border-t border-gray-200 bg-white p-4">
          <div className="max-w-4xl mx-auto">
            {/* Filters Expander */}
            {(countries.length > 0 || domains.length > 0 || requestedFields.length > 0 || limit) && (
              <div className="mb-3 flex flex-wrap gap-2">
                {countries.map(c => (
                  <span key={c} className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs flex items-center gap-2">
                    {c}
                    <button onClick={() => toggleArrayFilter(countries, setCountries, c)}>
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
                {domains.map(d => (
                  <span key={d} className="px-3 py-1 bg-green-50 text-green-700 rounded-full text-xs flex items-center gap-2">
                    {d}
                    <button onClick={() => toggleArrayFilter(domains, setDomains, d)}>
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
                <button
                  onClick={clearFilters}
                  className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-xs hover:bg-gray-200 transition-colors"
                >
                  Clear all
                </button>
              </div>
            )}

            <div className="flex gap-3">
              <div className="flex-1 bg-gray-100 rounded-xl p-3 focus-within:ring-2 focus-within:ring-blue-500 transition-shadow">
                <textarea
                  ref={textareaRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSearch();
                    }
                  }}
                  placeholder="Describe the candidates you're looking for..."
                  className="w-full bg-transparent resize-none outline-none text-gray-900 placeholder-gray-500"
                  rows="1"
                  style={{ maxHeight: '200px' }}
                />
                <div className="flex items-center gap-2 mt-2">
                  <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeAnswer}
                      onChange={(e) => setIncludeAnswer(e.target.checked)}
                      className="w-3.5 h-3.5 rounded border-gray-300"
                    />
                    <span>Include AI summary</span>
                  </label>
                </div>
              </div>
              
              <button
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-xl transition-colors flex items-center gap-2 font-medium"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Generate
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;