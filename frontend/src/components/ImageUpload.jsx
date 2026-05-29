import { useState, useRef } from 'react'

const ImageUpload = ({ onUpload }) => {
  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    if (selectedFile.type.startsWith('image/')) {
      setFile(selectedFile)
      // Create preview
      const reader = new FileReader()
      reader.onload = () => {
        setPreviewUrl(reader.result)
      }
      reader.readAsDataURL(selectedFile)
    } else {
      alert("Please select a valid image file")
    }
  }

  const onButtonClick = () => {
    inputRef.current.click()
  }

  const handleSubmit = () => {
    if (file) {
      onUpload(file)
    }
  }

  return (
    <div className="upload-wrapper card glass">
      <div 
        className={`upload-area ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        <input 
          ref={inputRef}
          type="file" 
          className="file-input" 
          accept="image/jpeg, image/png, image/jpg"
          onChange={handleChange}
        />
        
        <div className="upload-icon">📸</div>
        <div className="upload-text">Drag & drop your image here</div>
        <div className="upload-subtext">or click to browse from your device</div>
        
        {previewUrl && (
          <div className="selected-preview animate-fade-in" onClick={(e) => e.stopPropagation()}>
            <img src={previewUrl} alt="Preview" />
            <div style={{ marginTop: '1rem', color: 'hsl(var(--success))' }}>Image Selected: {file.name}</div>
          </div>
        )}
      </div>

      <div style={{ textAlign: 'center' }}>
        <button 
          className="btn-primary" 
          disabled={!file} 
          onClick={handleSubmit}
        >
          Generate XAI Prediction
        </button>
      </div>
    </div>
  )
}

export default ImageUpload
