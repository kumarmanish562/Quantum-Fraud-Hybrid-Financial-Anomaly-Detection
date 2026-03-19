import React, { useState } from 'react';
import { Card } from './ui';

const HelpSupport = () => {
  const [activeTab, setActiveTab] = useState('faq');
  const [ticketForm, setTicketForm] = useState({
    subject: '',
    category: 'Technical Issue',
    priority: 'Medium',
    description: ''
  });

  const faqs = [
    {
      question: 'How do I interpret fraud scores?',
      answer: 'Fraud scores range from 0-100%. Scores above 70% indicate high risk, 30-70% medium risk, and below 30% low risk. The quantum model provides more accurate scoring than classical models.'
    },
    {
      question: 'What should I do when I receive a fraud alert?',
      answer: 'Immediately review the transaction details, check the fraud score, and verify the transaction with the customer if needed. Use the investigation tools to gather more context before making a decision.'
    },
    {
      question: 'How often are the fraud detection models updated?',
      answer: 'Our quantum models are updated in real-time with new fraud patterns. Classical models are retrained weekly with the latest transaction data to maintain accuracy.'
    },
    {
      question: 'Can I customize alert thresholds?',
      answer: 'Yes, you can adjust fraud score thresholds in the Settings page. Higher thresholds reduce false positives but may miss some fraud cases.'
    },
    {
      question: 'How do I generate custom reports?',
      answer: 'Use the Report Generation page to create custom reports. You can filter by date range, transaction type, amount, and model type. Reports can be exported as PDF or CSV.'
    }
  ];

  const resources = [
    {
      title: 'User Guide',
      description: 'Complete guide to using the fraud detection system',
      icon: '📖',
      link: '#'
    },
    {
      title: 'API Documentation',
      description: 'Technical documentation for developers',
      icon: '🔧',
      link: '#'
    },
    {
      title: 'Video Tutorials',
      description: 'Step-by-step video guides and walkthroughs',
      icon: '🎥',
      link: '#'
    },
    {
      title: 'Best Practices',
      description: 'Industry best practices for fraud detection',
      icon: '⭐',
      link: '#'
    }
  ];

  const handleTicketSubmit = (e) => {
    e.preventDefault();
    console.log('Support ticket submitted:', ticketForm);
    alert('Support ticket submitted successfully! We\'ll get back to you within 24 hours.');
    setTicketForm({
      subject: '',
      category: 'Technical Issue',
      priority: 'Medium',
      description: ''
    });
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Help & Support</h1>
        <p className="text-gray-400">Get help with the fraud detection system and contact our support team</p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="text-center cursor-pointer hover:border-blue-500/30" onClick={() => setActiveTab('contact')}>
          <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <h3 className="text-white font-semibold mb-2">Contact Support</h3>
          <p className="text-gray-400 text-sm">Get help from our expert team</p>
        </Card>

        <Card className="text-center cursor-pointer hover:border-green-500/30">
          <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-white font-semibold mb-2">System Status</h3>
          <p className="text-green-400 text-sm">All systems operational</p>
        </Card>

        <Card className="text-center cursor-pointer hover:border-purple-500/30">
          <div className="w-16 h-16 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <h3 className="text-white font-semibold mb-2">Documentation</h3>
          <p className="text-gray-400 text-sm">Browse guides and tutorials</p>
        </Card>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-800/30 rounded-2xl p-1">
        {[
          { id: 'faq', label: 'FAQ' },
          { id: 'resources', label: 'Resources' },
          { id: 'contact', label: 'Contact Support' }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-3 px-4 rounded-xl font-medium transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-blue-500 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* FAQ Tab */}
      {activeTab === 'faq' && (
        <Card>
          <h2 className="text-xl font-semibold text-white mb-6">Frequently Asked Questions</h2>
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div key={index} className="border-b border-gray-800/50 pb-4 last:border-b-0">
                <h3 className="text-white font-medium mb-2">{faq.question}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{faq.answer}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Resources Tab */}
      {activeTab === 'resources' && (
        <div className="space-y-6">
          <Card>
            <h2 className="text-xl font-semibold text-white mb-6">Documentation & Resources</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {resources.map((resource, index) => (
                <div key={index} className="flex items-center space-x-4 p-4 bg-gray-800/30 rounded-xl hover:bg-gray-800/50 transition-colors duration-200 cursor-pointer">
                  <div className="text-2xl">{resource.icon}</div>
                  <div className="flex-1">
                    <h3 className="text-white font-medium">{resource.title}</h3>
                    <p className="text-gray-400 text-sm">{resource.description}</p>
                  </div>
                  <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <h2 className="text-xl font-semibold text-white mb-6">Contact Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-white font-medium mb-4">Support Hours</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Monday - Friday</span>
                    <span className="text-white">9:00 AM - 6:00 PM EST</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Saturday</span>
                    <span className="text-white">10:00 AM - 4:00 PM EST</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sunday</span>
                    <span className="text-white">Closed</span>
                  </div>
                </div>
              </div>
              <div>
                <h3 className="text-white font-medium mb-4">Emergency Contact</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="text-gray-400">Phone:</span>
                    <span className="text-white">+1 (800) 123-FRAUD</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-gray-400">Email:</span>
                    <span className="text-white">emergency@quantumfraud.com</span>
                  </div>
                  <p className="text-yellow-400 text-xs mt-2">Available 24/7 for critical security issues</p>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Contact Support Tab */}
      {activeTab === 'contact' && (
        <Card>
          <h2 className="text-xl font-semibold text-white mb-6">Submit Support Ticket</h2>
          <form onSubmit={handleTicketSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">Subject</label>
                <input
                  type="text"
                  value={ticketForm.subject}
                  onChange={(e) => setTicketForm(prev => ({ ...prev, subject: e.target.value }))}
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50"
                  placeholder="Brief description of your issue"
                  required
                />
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">Category</label>
                <select
                  value={ticketForm.category}
                  onChange={(e) => setTicketForm(prev => ({ ...prev, category: e.target.value }))}
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50"
                >
                  <option value="Technical Issue">Technical Issue</option>
                  <option value="Account Problem">Account Problem</option>
                  <option value="Feature Request">Feature Request</option>
                  <option value="Bug Report">Bug Report</option>
                  <option value="General Question">General Question</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">Priority</label>
                <select
                  value={ticketForm.priority}
                  onChange={(e) => setTicketForm(prev => ({ ...prev, priority: e.target.value }))}
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50"
                >
                  <option value="Low">Low</option>
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                  <option value="Critical">Critical</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">Description</label>
              <textarea
                value={ticketForm.description}
                onChange={(e) => setTicketForm(prev => ({ ...prev, description: e.target.value }))}
                rows={6}
                className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 resize-none"
                placeholder="Please provide detailed information about your issue..."
                required
              />
            </div>

            <div className="flex items-center space-x-4">
              <button
                type="submit"
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-xl font-medium hover:from-blue-700 hover:to-cyan-600 transition-all duration-200"
              >
                Submit Ticket
              </button>
              <p className="text-gray-400 text-sm">We typically respond within 24 hours</p>
            </div>
          </form>
        </Card>
      )}
    </div>
  );
};

export default HelpSupport;