document.addEventListener("DOMContentLoaded", () => {
  // —— Elements ——
  const dropArea = document.getElementById("drop-area");
  const browseBtn = document.getElementById("browse-btn");
  const fileInput = document.getElementById("file-input");
  const fileInfo = document.getElementById("file-info");
  const fileName = document.getElementById("file-name");
  const fileIcon = document.getElementById("file-icon");
  const fileTypeBadge = document.getElementById("file-type-badge");
  const errorMessage = document.getElementById("error-message");
  const instructionContent = document.getElementById("instruction-content");

  const sliderContainer = document.getElementById("slider-container");
  const intensitySlider = document.getElementById("intensity-slider");
  const intensityValue = document.getElementById("intensity-value");
  const chatForm = document.getElementById("chat-form");
  const formTitle = document.getElementById("form-title");
  const chatInput = document.getElementById("chat-input");
  const submitBtn = document.getElementById("submit-btn");
  const submitLoader = document.getElementById("submit-loader");
  const submitStatus = document.getElementById("submit-status");

  const sampleZipBtn = document.getElementById("sample-zip-btn");
  const samplePdfBtn = document.getElementById("sample-pdf-btn");
  const downloadFrame = document.getElementById("download-frame");

  // —— Constants ——
  const imgFormats = [
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
    "heic",
  ];
  const vidFormats = [
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
    "webm",
  ];
  const zipFormats = ["zip"];
  const pdfFormats = ["pdf"];

  // —— State ——
  let fileType = null;
  let isProcessing = false;
  let selectedFile = null;
  let processedFileId = null;

  // —— Endpoints ——
  let root = window.location.href;
  if (window.location.pathname === "/") {
    root = "";
  }
  const uploadEndpoint = `${root}/upload`;
  const blurEndpoint = `${root}/blur`;
  const redactEndpoint = `${root}/redact`;
  const downloadEndpoint = `${root}/download`;
  const sampleZipEndpoint = `${root}/sample-zip`;
  const samplePdfEndpoint = `${root}/sample-pdf`;

  // —— Utility ——
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  function highlight() {
    dropArea.classList.add("drag-over");
  }
  function unhighlight() {
    dropArea.classList.remove("drag-over");
  }

  function disableDropArea() {
    dropArea.classList.add("disabled");
    dropArea.style.pointerEvents = "none";
    dropArea.style.opacity = "0.5";
  }

  function enableDropArea() {
    dropArea.classList.remove("disabled");
    dropArea.style.pointerEvents = "auto";
    dropArea.style.opacity = "1";
  }

  function resetUI() {
    selectedFile = fileType = processedFileId = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    errorMessage.classList.add("hidden");

    instructionContent.classList.remove("hidden");
    sliderContainer.classList.add("hidden");
    chatForm.classList.add("hidden");

    submitBtn.disabled = true;
    submitBtn.textContent = "Submit";
    submitBtn.className = "primary-btn";
    submitLoader.style.display = "none";
    submitStatus.classList.add("hidden");
    setTimeout(() => {
      downloadFrame.src = "";
    }, 5000);
  }

  function setFileDisplay(file, type, ext) {
    fileName.textContent = file.name;
    switch (type) {
      case "pdf":
        fileIcon.innerHTML = PDF_SVG;
        break;
      case "zip":
        fileIcon.innerHTML = ZIP_SVG;
        break;
      case "img":
        fileIcon.innerHTML = IMG_SVG;
        break;
      case "vid":
        fileIcon.innerHTML = VID_SVG;
        break;
    }
    fileTypeBadge.textContent = ext.toUpperCase();
    fileTypeBadge.className = `file-type-badge ${type}`;
    fileInfo.classList.remove("hidden");
  }

  function showForm(type) {
    instructionContent.classList.add("hidden");
    chatForm.classList.remove("hidden");
    submitBtn.disabled = false;
    if (type === "pdf") {
      formTitle.textContent = "Redaction instructions";
      chatInput.placeholder = "Describe what to redact...";
    } else {
      sliderContainer.classList.remove("hidden");
      formTitle.textContent = "Label items you want to blur";
      chatInput.placeholder = "Document, monitor, person";
    }
  }

  async function uploadFile(file) {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch(uploadEndpoint, { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(`Upload failed: ${err.detail}`);
    }
    return (await resp.json()).fileId;
  }

  async function handleSubmit() {
    if (!selectedFile) return;
    // download state?
    if (submitBtn.classList.contains("success")) {
      downloadFrame.src = `${downloadEndpoint}/${processedFileId}`;
      setTimeout(resetUI, 200);
      return;
    }

    // upload
    disableDropArea();
    isProcessing = true;
    browseBtn.disabled = true;
    submitBtn.disabled = true;
    submitBtn.textContent = "Uploading…";
    submitLoader.style.display = "block";
    try {
      const fileId = await uploadFile(selectedFile);

      // process
      submitBtn.textContent = "Processing…";
      const fd = new FormData();
      fd.append("file_id", fileId);
      fd.append("prompt", chatInput.value.trim());
      if (fileType !== "pdf") {
        const intensityFloat = (
          parseInt(intensitySlider.value, 10) / 100
        ).toFixed(2);
        fd.append("intensity", intensityFloat);
      }

      const endpoint = fileType === "pdf" ? redactEndpoint : blurEndpoint;
      const resp = await fetch(endpoint, { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(`Processing failed: ${err.detail}`);
      }
      const result = await resp.json();
      processedFileId = result.fileId;

      // success
      enableDropArea();
      isProcessing = false;
      browseBtn.disabled = false;
      submitBtn.disabled = false;
      submitBtn.textContent = "Download";
      submitBtn.classList.add("success");
      submitStatus.textContent = "Done!";
      submitStatus.style.color = "var(--success-color)";
      submitLoader.style.display = "none";
      submitStatus.classList.remove("hidden");
    } catch (err) {
      submitBtn.textContent = "Submit";
      submitBtn.disabled = false;
      submitLoader.style.display = "none";
      submitStatus.textContent = err.message;
      submitStatus.style.color = "var(--error-color)";
      submitStatus.classList.remove("hidden");
    }
  }

  function validateAndProcessFile(file) {
    resetUI();
    const name = file.name.toLowerCase();
    const ext = name.split(".").pop();

    fileType = null;
    if (pdfFormats.includes(ext)) {
      fileType = "pdf";
    } else if (zipFormats.includes(ext)) {
      fileType = "zip";
    } else if (imgFormats.includes(ext)) {
      fileType = "img";
    } else if (vidFormats.includes(ext)) {
      fileType = "vid";
    }

    if (fileType) {
      selectedFile = file;
      setFileDisplay(file, fileType, ext);
      showForm(ext);
    } else {
      errorMessage.textContent = "Unsupported file format.";
      errorMessage.classList.remove("hidden");
    }
  }

  // —— Event Listeners ——
  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) =>
    validateAndProcessFile(e.target.files[0])
  );
  intensitySlider.addEventListener("input", () => {
    intensityValue.textContent = `${intensitySlider.value}%`;
  });

  ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
    dropArea.addEventListener(evt, preventDefaults, false);
    document.body.addEventListener(evt, preventDefaults, false);
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dropArea.addEventListener(evt, highlight, false)
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropArea.addEventListener(evt, unhighlight, false)
  );
  dropArea.addEventListener("drop", (e) => {
    if (isProcessing) return;
    validateAndProcessFile(e.dataTransfer.files[0]);
  });

  submitBtn.addEventListener("click", handleSubmit);

  sampleZipBtn.addEventListener("click", () => {
    downloadFrame.src = sampleZipEndpoint;
  });
  samplePdfBtn.addEventListener("click", () => {
    downloadFrame.src = samplePdfEndpoint;
  });

  // —— Inline SVGs ——
  const ZIP_SVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" 
        viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M22 12v9a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-9"/>
      <path d="M2 12V5a1 1 0 0 1 1-1h18a1 1 0 0 1 1 1v7"/>
      <path d="M12 12v9"/>
      <path d="M12 12 8 8"/>
      <path d="m12 12 4-4"/>
    </svg>`;
  const PDF_SVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" 
        viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>`;
  const VID_SVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" 
        viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
        stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="7" width="14" height="12" rx="2" ry="2"></rect>
      <polygon points="17 11 22 8 22 18 17 15 17 11"></polygon>
    </svg>`;
  const IMG_SVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" 
        viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
        stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
      <circle cx="8.5" cy="8.5" r="1.5"/>
      <path d="M21 15l-5-5L5 21"/>
    </svg>`;
});
