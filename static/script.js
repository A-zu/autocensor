document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropArea = document.getElementById("drop-area");
  const fileInfo = document.getElementById("file-info");
  const fileName = document.getElementById("file-name");
  const fileTypeBadge = document.getElementById("file-type-badge");
  const fileIcon = document.getElementById("file-icon");
  const browseBtn = document.getElementById("browse-btn");
  const fileInput = document.getElementById("file-input");
  const sampleBtn = document.getElementById("sample-btn");
  const samplePdfBtn = document.getElementById("sample-pdf-btn");
  const uploadForm = document.getElementById("upload-form");
  const errorMessage = document.getElementById("error-message");
  const downloadFrame = document.getElementById("download-frame");

  // Content sections
  const instructionContent = document.getElementById("instruction-content");
  const zipContent = document.getElementById("zip-content");
  const pdfContent = document.getElementById("pdf-content");

  // ZIP content elements
  const chatInput = document.getElementById("chat-input");
  const submitBtn = document.getElementById("submit-btn");
  const submitLoader = document.getElementById("submit-loader");
  const submitStatus = document.getElementById("submit-status");

  // PDF content elements
  const pdfChatInput = document.getElementById("pdf-chat-input");
  const pdfSubmitBtn = document.getElementById("pdf-submit-btn");
  const pdfSubmitLoader = document.getElementById("pdf-submit-loader");
  const pdfSubmitStatus = document.getElementById("pdf-submit-status");

  // State variables
  let selectedFile = null;
  let processedFileId = null;
  let fileType = null;

  // Fix URL path if needed
  if (
    window.location.pathname.endsWith("/") &&
    window.location.pathname !== "/"
  ) {
    const newUrl =
      window.location.pathname.slice(0, -1) +
      window.location.search +
      window.location.hash;
    window.history.replaceState(null, "", newUrl);
  }

  // API endpoints
  let root = window.location.href;
  if (window.location.pathname === "/") {
    root = "";
  }

  const blurEndpoint = `${root}/blur`;
  const chatEndpoint = `${root}/chat`;
  const uploadEndpoint = `${root}/upload`;
  const redactEndpoint = `${root}/redact`;
  const downloadEndpoint = `${root}/download`;
  const sampleZipEndpoint = `${root}/sample-zip`;
  const samplePdfEndpoint = `${root}/sample-pdf`;

  // Multiselect elements
  const multiselectBtn = document.getElementById("multiselect-btn");
  const multiselectOptions = document.getElementById("multiselect-options");
  const multiselectText = document.getElementById("multiselect-text");
  const multiselectSearch = document.getElementById("multiselect-search");
  const multiselectItems = document.getElementById("multiselect-items");
  const multiselectSelected = document.getElementById("multiselect-selected");
  const multiselectArrow = multiselectBtn.querySelector(".multiselect-arrow");

  // Available items for the multiselect dropdown
  const items = [
    { id: 0, name: "person" },
    { id: 1, name: "bicycle" },
    { id: 2, name: "car" },
    { id: 3, name: "motorcycle" },
    { id: 4, name: "airplane" },
    { id: 5, name: "bus" },
    { id: 6, name: "train" },
    { id: 7, name: "truck" },
    { id: 8, name: "boat" },
    { id: 9, name: "traffic light" },
    { id: 10, name: "fire hydrant" },
    { id: 11, name: "stop sign" },
    { id: 12, name: "parking meter" },
    { id: 13, name: "bench" },
    { id: 14, name: "bird" },
    { id: 15, name: "cat" },
    { id: 16, name: "dog" },
    { id: 17, name: "horse" },
    { id: 18, name: "sheep" },
    { id: 19, name: "cow" },
    { id: 20, name: "elephant" },
    { id: 21, name: "bear" },
    { id: 22, name: "zebra" },
    { id: 23, name: "giraffe" },
    { id: 24, name: "backpack" },
    { id: 25, name: "umbrella" },
    { id: 26, name: "handbag" },
    { id: 27, name: "tie" },
    { id: 28, name: "suitcase" },
    { id: 29, name: "frisbee" },
    { id: 30, name: "skis" },
    { id: 31, name: "snowboard" },
    { id: 32, name: "sports ball" },
    { id: 33, name: "kite" },
    { id: 34, name: "baseball bat" },
    { id: 35, name: "baseball glove" },
    { id: 36, name: "skateboard" },
    { id: 37, name: "surfboard" },
    { id: 38, name: "tennis racket" },
    { id: 39, name: "bottle" },
    { id: 40, name: "wine glass" },
    { id: 41, name: "cup" },
    { id: 42, name: "fork" },
    { id: 43, name: "knife" },
    { id: 44, name: "spoon" },
    { id: 45, name: "bowl" },
    { id: 46, name: "banana" },
    { id: 47, name: "apple" },
    { id: 48, name: "sandwich" },
    { id: 49, name: "orange" },
    { id: 50, name: "broccoli" },
    { id: 51, name: "carrot" },
    { id: 52, name: "hot dog" },
    { id: 53, name: "pizza" },
    { id: 54, name: "donut" },
    { id: 55, name: "cake" },
    { id: 56, name: "chair" },
    { id: 57, name: "couch" },
    { id: 58, name: "potted plant" },
    { id: 59, name: "bed" },
    { id: 60, name: "dining table" },
    { id: 61, name: "toilet" },
    { id: 62, name: "tv" },
    { id: 63, name: "laptop" },
    { id: 64, name: "mouse" },
    { id: 65, name: "remote" },
    { id: 66, name: "keyboard" },
    { id: 67, name: "cell phone" },
    { id: 68, name: "microwave" },
    { id: 69, name: "oven" },
    { id: 70, name: "toaster" },
    { id: 71, name: "sink" },
    { id: 72, name: "refrigerator" },
    { id: 73, name: "book" },
    { id: 74, name: "clock" },
    { id: 75, name: "vase" },
    { id: 76, name: "scissors" },
    { id: 77, name: "teddy bear" },
    { id: 78, name: "hair drier" },
    { id: 79, name: "toothbrush" },
    { id: 80, name: "text" },
  ];

  // Selected item IDs - set default selections
  let selectedItemIds = [62, 63, 80]; // tv, laptop, text

  // Initialize multiselect dropdown
  function initMultiselect() {
    // Populate options
    renderMultiselectOptions();

    // Update selected items display for default selections
    updateSelectedItems();

    // Toggle dropdown on button click
    multiselectBtn.addEventListener("click", () => {
      multiselectOptions.classList.toggle("show");
      multiselectArrow.classList.toggle("up");
      if (multiselectOptions.classList.contains("show")) {
        multiselectSearch.focus();
      }
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".multiselect-dropdown")) {
        multiselectOptions.classList.remove("show");
        multiselectArrow.classList.remove("up");
      }
    });

    // Search functionality
    multiselectSearch.addEventListener("input", () => {
      const searchTerm = multiselectSearch.value.toLowerCase();
      renderMultiselectOptions(searchTerm);
    });
  }

  // Render multiselect options
  function renderMultiselectOptions(searchTerm = "") {
    multiselectItems.innerHTML = "";

    items.forEach((item) => {
      if (searchTerm === "" || item.name.toLowerCase().includes(searchTerm)) {
        const option = document.createElement("div");
        option.className = "multiselect-option";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.id = `item-${item.id}`;
        checkbox.checked = selectedItemIds.includes(item.id);

        const label = document.createElement("label");
        label.htmlFor = `item-${item.id}`;
        label.textContent = item.name;

        option.appendChild(checkbox);
        option.appendChild(label);

        option.addEventListener("click", (e) => {
          if (e.target !== checkbox) {
            checkbox.checked = !checkbox.checked;
          }

          if (checkbox.checked) {
            if (!selectedItemIds.includes(item.id)) {
              selectedItemIds.push(item.id);
            }
          } else {
            selectedItemIds = selectedItemIds.filter((id) => id !== item.id);
          }

          updateSelectedItems();
        });

        multiselectItems.appendChild(option);
      }
    });
  }

  // Update selected items display
  function updateSelectedItems() {
    multiselectSelected.innerHTML = "";

    if (selectedItemIds.length === 0) {
      multiselectText.textContent = "Select items...";
      multiselectText.className = "multiselect-placeholder";
      return;
    }

    multiselectText.textContent = `${selectedItemIds.length} item(s) selected`;
    multiselectText.className = "";

    selectedItemIds.forEach((id) => {
      const item = items.find((item) => item.id === id);
      if (!item) return;

      const tag = document.createElement("div");
      tag.className = "multiselect-tag";
      tag.textContent = item.name;

      const removeBtn = document.createElement("span");
      removeBtn.className = "multiselect-tag-remove";
      removeBtn.textContent = "×";
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedItemIds = selectedItemIds.filter((itemId) => itemId !== id);
        updateSelectedItems();
        renderMultiselectOptions(multiselectSearch.value.toLowerCase());
      });

      tag.appendChild(removeBtn);
      multiselectSelected.appendChild(tag);
    });
  }

  // Handle browse button click
  browseBtn.addEventListener("click", () => {
    fileInput.click();
  });

  // Handle file selection via input
  fileInput.addEventListener("change", handleFileSelect);

  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight drop area when item is dragged over it
  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  // Remove highlight when item is dragged out or dropped
  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  // Handle dropped files
  dropArea.addEventListener("drop", handleDrop, false);

  // Handle sample button clicks
  sampleBtn.addEventListener("click", () => {
    downloadFrame.src = sampleZipEndpoint;
  });

  samplePdfBtn.addEventListener("click", () => {
    downloadFrame.src = samplePdfEndpoint;
  });

  // Handle submit button clicks
  submitBtn.addEventListener("click", handleZipSubmit);
  pdfSubmitBtn.addEventListener("click", handlePdfSubmit);

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

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
      validateAndProcessFile(files[0]);
    }
  }

  function handleFileSelect(e) {
    const files = e.target.files;

    if (files.length > 0) {
      validateAndProcessFile(files[0]);
    }
  }

  function validateAndProcessFile(file) {
    // Reset UI elements
    errorMessage.classList.add("hidden");
    errorMessage.textContent = "";
    resetSubmitStates();

    // Check file type
    const lowerCaseName = file.name.toLowerCase();

    if (lowerCaseName.endsWith(".zip") || lowerCaseName.endsWith(".pdf")) {
      // Store the selected file
      selectedFile = file;

      // Determine file type
      fileType = lowerCaseName.endsWith(".zip") ? "zip" : "pdf";

      // Update file icon based on type
      if (fileType === "zip") {
        fileIcon.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12v9a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-9"/>
          <path d="M2 12V5a1 1 0 0 1 1-1h18a1 1 0 0 1 1 1v7"/>
          <path d="M12 12v9"/>
          <path d="M12 12 8 8"/>
          <path d="m12 12 4-4"/>
        </svg>
      `;
      } else {
        fileIcon.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
      `;
      }

      // Update file type badge
      fileTypeBadge.textContent = fileType.toUpperCase();
      fileTypeBadge.className = `file-type-badge ${fileType}`;

      // Display file info
      fileName.textContent = file.name;
      fileInfo.classList.remove("hidden");

      // Show appropriate content section
      showContentSection(fileType);
    } else {
      // Show error for unsupported files
      errorMessage.textContent = "Please select a .zip or .pdf file";
      errorMessage.classList.remove("hidden");
      fileInfo.classList.add("hidden");
      selectedFile = null;
      fileType = null;
    }
  }

  function showContentSection(type) {
    // Hide all content sections
    instructionContent.classList.add("hidden");
    zipContent.classList.add("hidden");
    pdfContent.classList.add("hidden");

    // Show appropriate section
    if (type === "zip") {
      zipContent.classList.remove("hidden");
      submitBtn.disabled = false;
    } else if (type === "pdf") {
      pdfContent.classList.remove("hidden");
      pdfSubmitBtn.disabled = false;
    }
  }

  function resetSubmitStates() {
    // Reset ZIP submit button
    submitBtn.disabled = true;
    submitBtn.textContent = "Submit";
    submitBtn.className = "primary-btn";
    submitLoader.style.display = "none";
    submitStatus.classList.add("hidden");

    // Reset PDF submit button
    pdfSubmitBtn.disabled = true;
    pdfSubmitBtn.textContent = "Submit";
    pdfSubmitBtn.className = "primary-btn";
    pdfSubmitLoader.style.display = "none";
    pdfSubmitStatus.classList.add("hidden");
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(uploadEndpoint, {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      return data.fileId;
    } else {
      const data = await response.json();
      throw new Error(`Upload failed: ${data.detail}`);
    }
  }

  async function handleZipSubmit() {
    if (!selectedFile) return;

    submitStatus.classList.add("hidden");

    // Disable file changes
    dropArea.style.pointerEvents = "none";
    browseBtn.disabled = true;

    // If button is in download state, trigger download
    if (submitBtn.classList.contains("success")) {
      initiateDownload(processedFileId);
      return;
    }

    try {
      // Step 1: Chat processing
      submitBtn.textContent = "Thinking...";
      submitBtn.disabled = true;
      submitLoader.style.display = "block";

      const chatMessage = chatInput.value.trim();

      if (chatMessage) {
        const formData = new FormData();
        formData.append("prompt", chatMessage);

        const chatResponse = await fetch(chatEndpoint, {
          method: "POST",
          body: formData,
        });

        if (chatResponse.ok) {
          const chatData = await chatResponse.json();

          // Update message
          submitStatus.textContent = chatData.message;

          // Update selected items if provided
          if (
            chatData.selectedItemIds &&
            Array.isArray(chatData.selectedItemIds)
          ) {
            selectedItemIds = chatData.selectedItemIds;
            updateSelectedItems();
            renderMultiselectOptions();
          }
        }
      }

      // Step 2: Upload file
      submitBtn.textContent = "Uploading...";
      submitBtn.disabled = true;
      submitLoader.style.display = "block";

      const fileId = await uploadFile(selectedFile);

      // Step 3: File upload processing
      submitBtn.textContent = "Processing...";

      const formData = new FormData();
      formData.append("file_id", fileId);
      formData.append("selectedItemIds", JSON.stringify(selectedItemIds));

      const response = await fetch(blurEndpoint, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        processedFileId = fileId;

        // Success state
        submitBtn.textContent = "Download";
        submitBtn.className = "primary-btn success";
        submitBtn.disabled = false;
        submitLoader.style.display = "none";

        submitStatus.textContent = data.message;
        submitStatus.style.color = "var(--success-color)"; // ✅ reset color
        submitStatus.classList.remove("hidden");
      } else {
        const data = await response.json();
        throw new Error(`Processing failed: ${data.detail}`);
      }
    } catch (error) {
      submitBtn.textContent = "Submit";
      submitBtn.disabled = false;
      submitLoader.style.display = "none";

      submitStatus.textContent = error.message;
      submitStatus.style.color = "var(--error-color)";
      submitStatus.classList.remove("hidden");
    } finally {
      // Re-enable file changes
      dropArea.style.pointerEvents = "auto";
      browseBtn.disabled = false;
    }
  }

  async function handlePdfSubmit() {
    if (!selectedFile) return;

    pdfSubmitStatus.classList.add("hidden");

    // Disable file changes
    dropArea.style.pointerEvents = "none";
    browseBtn.disabled = true;

    // If button is in download state, trigger download
    if (pdfSubmitBtn.classList.contains("success")) {
      initiateDownload(processedFileId);
      return;
    }

    try {
      // Step 1: Upload file
      pdfSubmitBtn.textContent = "Uploading...";
      pdfSubmitBtn.disabled = true;
      pdfSubmitLoader.style.display = "block";

      const fileId = await uploadFile(selectedFile);

      // Step 2: PDF processing
      pdfSubmitBtn.textContent = "Processing...";

      const formData = new FormData();
      formData.append("file_id", fileId);
      formData.append("prompt", pdfChatInput.value.trim());

      const response = await fetch(redactEndpoint, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        // Set file id to download
        processedFileId = fileId;

        // Success state
        pdfSubmitBtn.textContent = "Download";
        pdfSubmitBtn.className = "primary-btn success";
        pdfSubmitBtn.disabled = false;
        pdfSubmitLoader.style.display = "none";

        pdfSubmitStatus.textContent =
          data.message || "File processed successfully";
        pdfSubmitStatus.style.color = "var(--success-color)"; // ✅ reset color
        pdfSubmitStatus.classList.remove("hidden");
      } else {
        const data = await response.json();
        throw new Error(`Processing failed: ${data.detail}`);
      }
    } catch (error) {
      pdfSubmitBtn.textContent = "Submit";
      pdfSubmitBtn.disabled = false;
      pdfSubmitLoader.style.display = "none";

      pdfSubmitStatus.textContent = error.message;
      pdfSubmitStatus.style.color = "var(--error-color)";
      pdfSubmitStatus.classList.remove("hidden");
    } finally {
      // Re-enable file changes
      dropArea.style.pointerEvents = "auto";
      browseBtn.disabled = false;
    }
  }

  function initiateDownload(fileId) {
    downloadFrame.src = `${downloadEndpoint}/${fileId}`;

    // Reset UI after download
    setTimeout(() => {
      resetUI();
    }, 100);
  }

  function resetUI() {
    // Reset file state
    selectedFile = null;
    fileType = null;
    processedFileId = null;

    // Clear file input
    fileInput.value = "";
    fileInfo.classList.add("hidden");

    // Re-enable file changes
    dropArea.style.pointerEvents = "auto";
    browseBtn.disabled = false;

    // Show instruction content
    instructionContent.classList.remove("hidden");
    zipContent.classList.add("hidden");
    pdfContent.classList.add("hidden");

    // Clear inputs
    chatInput.value = "";
    pdfChatInput.value = "";

    // Reset submit states
    resetSubmitStates();
  }

  // Initialize the multiselect dropdown
  initMultiselect();
});
