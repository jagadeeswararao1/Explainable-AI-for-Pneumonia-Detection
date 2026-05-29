import { useEffect, useState } from 'react'

const ResultsDisplay = ({ data }) => {
  const { prediction, confidence, original_image, gradcam_image, lime_image, saliency_map, combined_xai } = data
  const [barWidth, setBarWidth] = useState(0)

  useEffect(() => {
    // Animate the confidence bar on mount
    const timer = setTimeout(() => {
      setBarWidth(confidence)
    }, 100)
    return () => clearTimeout(timer)
  }, [confidence])

  return (
    <div className="results-container">
      {/* Overview Card */}
      <div className="results-header card glass">
        <div>
          <div style={{ color: 'hsl(var(--text-muted))', fontSize: '1.1rem' }}>Model Prediction</div>
          <div className="prediction-class">{prediction}</div>
        </div>

        <div className="confidence-bar-container">
          <div className="confidence-label">
            <span>Confidence</span>
            <span style={{ color: 'hsl(var(--primary))', fontWeight: 'bold' }}>{confidence.toFixed(2)}%</span>
          </div>
          <div className="confidence-track">
            <div
              className="confidence-fill"
              style={{ width: `${barWidth}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Images Grid */}
      <div className="images-grid">
        <div className="image-card card glass">
          <h3>Original Input</h3>
          {original_image ? (
            <img src={original_image} alt="Original Input" className="img-preview" />
          ) : (
            <div className="img-preview" style={{ background: '#111', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>No Image</div>
          )}
        </div>

        <div className="image-card card glass" style={{ borderColor: 'rgba(250, 100%, 75%, 0.3)' }}>
          <h3>Grad-CAM++</h3>
          <p style={{ fontSize: '0.85rem', color: 'hsl(var(--text-muted))', marginBottom: '1rem', textAlign: 'center' }}>
            Heatmap highlighting critical regions for classification.
          </p>
          {gradcam_image ? (
            <img src={gradcam_image} alt="Grad-CAM++" className="img-preview" />
          ) : (
            <div className="img-preview" style={{ background: '#111', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>N/A</div>
          )}
        </div>

        <div className="image-card card glass" style={{ borderColor: 'rgba(190, 90%, 65%, 0.3)' }}>
          <h3>LIME Superpixels</h3>
          <p style={{ fontSize: '0.85rem', color: 'hsl(var(--text-muted))', marginBottom: '1rem', textAlign: 'center' }}>
            Boundary mapping of most influential local features.
          </p>
          {lime_image ? (
            <img src={lime_image} alt="LIME" className="img-preview" />
          ) : (
            <div className="img-preview" style={{ background: '#111', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>N/A</div>
          )}
        </div>

        <div className="image-card card glass" style={{ borderColor: 'rgba(142, 71%, 45%, 0.3)' }}>
          <h3>Saliency Map</h3>
          <p style={{ fontSize: '0.85rem', color: 'hsl(var(--text-muted))', marginBottom: '1rem', textAlign: 'center' }}>
            Pixel-level gradient emphasis for classification.
          </p>
          {saliency_map ? (
            <img src={saliency_map} alt="Saliency Map" className="img-preview" />
          ) : (
            <div className="img-preview" style={{ background: '#111', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>N/A</div>
          )}
        </div>

        <div className="image-card card glass" style={{ borderColor: 'rgba(250, 200%, 75%, 0.3)' }}>
          <h3>Combined XAI</h3>
          <p style={{ fontSize: '0.85rem', color: 'hsl(var(--text-muted))', marginBottom: '1rem', textAlign: 'center' }}>
            Grad-CAM and Saliency overlay combined.
          </p>
          {combined_xai ? (
            <img src={combined_xai} alt="Combined XAI" className="img-preview" />
          ) : (
            <div className="img-preview" style={{ background: '#111', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>N/A</div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ResultsDisplay
