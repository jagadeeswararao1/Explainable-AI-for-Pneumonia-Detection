import { useState } from 'react'
import './App.css'
import ImageUpload from './components/ImageUpload'
import ResultsDisplay from './components/ResultsDisplay'

function App() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleProcessImage = async (file) => {
    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('image', file)

    try {
      // Assuming backend is running on 8000
      const response = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError('Failed to process image. Make sure the backend limit is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <header className="header animate-fade-in">
        <div className="container">
          <h1><span className="text-gradient">XAI</span> Dashboard</h1>
          <p>Explainable AI for PyTorch EfficientNet using Grad-CAM++ & LIME</p>
        </div>
      </header>

      <main className="main-content container">
        {error && (
          <div className="card glass animate-fade-in" style={{ backgroundColor: 'rgba(255,0,0,0.1)', borderColor: 'red' }}>
            <p style={{ color: 'var(--danger)', textAlign: 'center' }}>{error}</p>
          </div>
        )}

        {!loading && !results && (
          <div className="animate-fade-in">
            <ImageUpload onUpload={handleProcessImage} />
          </div>
        )}

        {loading && (
          <div className="loader-container glass card animate-fade-in">
            <div className="spinner"></div>
            <h3>Running Interference & Extracting XAI Maps...</h3>
            <p style={{ color: 'hsl(var(--text-muted))', marginTop: '1rem' }}>This may take a few moments.</p>
          </div>
        )}

        {results && !loading && (
          <div className="animate-fade-in">
             <div style={{display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem'}}>
               <button className="btn-primary" onClick={handleReset} style={{marginTop: 0, padding: '0.5rem 1rem'}}>
                 Test Another Image
               </button>
             </div>
             <ResultsDisplay data={results} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
