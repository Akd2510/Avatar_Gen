import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [templates, setTemplates] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("");
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploadingTemplate, setIsUploadingTemplate] = useState(false);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/templates`);
      const fetchedTemplates = response.data.templates;
      setTemplates(fetchedTemplates);
      if (fetchedTemplates.length > 0 && !selectedTemplate) {
        setSelectedTemplate(fetchedTemplates[0]);
      }
    } catch (err) {
      console.error("Error fetching templates:", err);
      setError("Failed to load templates.");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setInputImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultUrl(null);
      setError(null);
    }
  };

  const handleCustomTemplateUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const formData = new FormData();
      formData.append("template_image", file);

      setIsUploadingTemplate(true);
      try {
        const response = await axios.post(
          `${API_BASE_URL}/api/upload-template`,
          formData,
        );
        const newTemplate = response.data.template_name;
        setTemplates((prev) => [newTemplate, ...prev]);
        setSelectedTemplate(newTemplate);
      } catch (err) {
        setError("Template upload failed.");
      } finally {
        setIsUploadingTemplate(false);
      }
    }
  };

  const handleSwap = async () => {
    if (!inputImage || !selectedTemplate) {
      setError("Please provide both input and template.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResultUrl(null);

    const formData = new FormData();
    formData.append("input_image", inputImage);
    formData.append("template_name", selectedTemplate);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/swap`, formData);
      setResultUrl(`${API_BASE_URL}${response.data.output_url}`);
    } catch (err: any) {
      console.error("Swap failed:", err);
      setError(
        err.response?.data?.detail || "Processing failed. Please try again.",
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header className="main-header">
        <h1>Avatar Generator</h1>
      </header>

      <div className="workflow-grid">
        <div className="input-column">
          <section className="panel">
            <div className="panel-info">
              <span className="count">01</span>
              <h3>Input Image</h3>
            </div>
            <div className={`upload-box ${previewUrl ? "active" : ""}`}>
              <input
                type="file"
                id="file-input"
                accept="image/*"
                onChange={handleFileChange}
              />
              <label htmlFor="file-input">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="preview-img" />
                ) : (
                  <div className="placeholder">
                    <div className="icon">+</div>
                    <span>Select Source Face</span>
                  </div>
                )}
              </label>
            </div>
          </section>

          <section className="action-panel">
            {error && <div className="error-alert">{error}</div>}
            <button
              className={`action-button ${isProcessing ? "processing" : ""}`}
              onClick={handleSwap}
              disabled={isProcessing || !inputImage}
            >
              {isProcessing ? "Processing..." : "Run Swapping"}
            </button>
          </section>
        </div>

        <div className="template-column">
          <section className="panel">
            <div className="panel-info">
              <span className="count">02</span>
              <h3>Target Template</h3>
            </div>

            <div className="templates-container">
              <div className="template-item add-new">
                <input
                  type="file"
                  id="template-upload"
                  accept="image/*"
                  onChange={handleCustomTemplateUpload}
                />
                <label htmlFor="template-upload">
                  <span>{isUploadingTemplate ? "..." : "+"}</span>
                </label>
              </div>

              {templates.map((template) => (
                <div
                  key={template}
                  className={`template-item ${selectedTemplate === template ? "active" : ""}`}
                  onClick={() => setSelectedTemplate(template)}
                >
                  <img
                    src={`${API_BASE_URL}/templates/${template}`}
                    alt={template}
                  />
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>

      {resultUrl && (
        <div className="modal-backdrop">
          <div className="modal-content">
            <button
              className="close-trigger"
              onClick={() => setResultUrl(null)}
            >
              Close
            </button>
            <div className="image-frame">
              <img src={resultUrl} alt="Swap Result" />
            </div>
            <div className="modal-actions">
              <a
                href={resultUrl}
                download="result.jpg"
                className="primary-link"
              >
                Download Image
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
