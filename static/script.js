document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const loader = document.getElementById("loader")
  const dropArea = document.getElementById("drop-area")
  const fileInfo = document.getElementById("file-info")
  const fileName = document.getElementById("file-name")
  const fileTypeBadge = document.getElementById("file-type-badge")
  const fileIcon = document.getElementById("file-icon")
  const actionBtn = document.getElementById("action-btn")
  const browseBtn = document.getElementById("browse-btn")
  const fileInput = document.getElementById("file-input")
  const exampleBtn = document.getElementById("example-btn")
  const examplePdfBtn = document.getElementById("example-pdf-btn")
  const uploadForm = document.getElementById("upload-form")
  const errorMessage = document.getElementById("error-message")
  const downloadFrame = document.getElementById("download-frame")
  const statusMessage = document.getElementById("status-message")

  // Chat elements
  const chatInput = document.getElementById("chat-input")
  const chatSubmit = document.getElementById("chat-submit")
  const chatLoader = document.getElementById("chat-loader")
  const chatStatusText = document.getElementById("chat-status-text")

  // State variables
  let selectedFile = null
  let processedFileId = null
  let fileType = null

  // Fix URL path if needed
  if (window.location.pathname.endsWith("/") && window.location.pathname !== "/") {
    const newUrl = window.location.pathname.slice(0, -1) + window.location.search + window.location.hash
    window.history.replaceState(null, "", newUrl)
  }

  // API endpoints
  let root = window.location.href
  if (window.location.pathname === "/") {
    root = ""
  }

  const uploadEndpoint = `${root}/upload`
  const redactEndpoint = `${root}/redact`
  const downloadEndpoint = `${root}/download`
  const exampleZipEndpoint = `${root}/example-zip`
  const examplePdfEndpoint = `${root}/example-pdf`

  // Multiselect elements
  const multiselectBtn = document.getElementById("multiselect-btn")
  const multiselectOptions = document.getElementById("multiselect-options")
  const multiselectText = document.getElementById("multiselect-text")
  const multiselectSearch = document.getElementById("multiselect-search")
  const multiselectItems = document.getElementById("multiselect-items")
  const multiselectSelected = document.getElementById("multiselect-selected")
  const multiselectArrow = multiselectBtn.querySelector(".multiselect-arrow")

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
  ]

  // Selected item IDs - set default selections
  let selectedItemIds = [62, 63, 80] // tv, laptop, text

  // Initialize multiselect dropdown
  function initMultiselect() {
    // Populate options
    renderMultiselectOptions()

    // Update selected items display for default selections
    updateSelectedItems()

    // Toggle dropdown on button click
    multiselectBtn.addEventListener("click", () => {
      multiselectOptions.classList.toggle("show")
      multiselectArrow.classList.toggle("up")
      if (multiselectOptions.classList.contains("show")) {
        multiselectSearch.focus()
      }
    })

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".multiselect-dropdown")) {
        multiselectOptions.classList.remove("show")
        multiselectArrow.classList.remove("up")
      }
    })

    // Search functionality
    multiselectSearch.addEventListener("input", () => {
      const searchTerm = multiselectSearch.value.toLowerCase()
      renderMultiselectOptions(searchTerm)
    })
  }

  // Render multiselect options
  function renderMultiselectOptions(searchTerm = "") {
    multiselectItems.innerHTML = ""

    items.forEach((item) => {
      if (searchTerm === "" || item.name.toLowerCase().includes(searchTerm)) {
        const option = document.createElement("div")
        option.className = "multiselect-option"

        const checkbox = document.createElement("input")
        checkbox.type = "checkbox"
        checkbox.id = `item-${item.id}`
        checkbox.checked = selectedItemIds.includes(item.id)

        const label = document.createElement("label")
        label.htmlFor = `item-${item.id}`
        label.textContent = item.name

        option.appendChild(checkbox)
        option.appendChild(label)

        option.addEventListener("click", (e) => {
          if (e.target !== checkbox) {
            checkbox.checked = !checkbox.checked
          }

          if (checkbox.checked) {
            if (!selectedItemIds.includes(item.id)) {
              selectedItemIds.push(item.id)
            }
          } else {
            selectedItemIds = selectedItemIds.filter((id) => id !== item.id)
          }

          updateSelectedItems()
        })

        multiselectItems.appendChild(option)
      }
    })
  }

  // Update selected items display
  function updateSelectedItems() {
    multiselectSelected.innerHTML = ""

    if (selectedItemIds.length === 0) {
      multiselectText.textContent = "Select items..."
      multiselectText.className = "multiselect-placeholder"
      return
    }

    multiselectText.textContent = `${selectedItemIds.length} item(s) selected`
    multiselectText.className = ""

    selectedItemIds.forEach((id) => {
      const item = items.find((item) => item.id === id)
      if (!item) return

      const tag = document.createElement("div")
      tag.className = "multiselect-tag"
      tag.textContent = item.name

      const removeBtn = document.createElement("span")
      removeBtn.className = "multiselect-tag-remove"
      removeBtn.textContent = "×"
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation()
        selectedItemIds = selectedItemIds.filter((itemId) => itemId !== id)
        updateSelectedItems()
        renderMultiselectOptions(multiselectSearch.value.toLowerCase())
      })

      tag.appendChild(removeBtn)
      multiselectSelected.appendChild(tag)
    })
  }

  // Handle browse button click
  browseBtn.addEventListener("click", () => {
    fileInput.click()
  })

  // Handle file selection via input
  fileInput.addEventListener("change", handleFileSelect)

  // Prevent default drag behaviors
  ;["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false)
    document.body.addEventListener(eventName, preventDefaults, false)
  })

  // Highlight drop area when item is dragged over it
  ;["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false)
  })

  // Remove highlight when item is dragged out or dropped
  ;["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false)
  })

  // Handle dropped files
  dropArea.addEventListener("drop", handleDrop, false)

  // Handle action button click (upload or download)
  actionBtn.addEventListener("click", handleActionButtonClick)

  // Handle example button click
  exampleBtn.addEventListener("click", () => {
    downloadFrame.src = exampleZipEndpoint
  })

  // Handle example PDF button click
  examplePdfBtn.addEventListener("click", () => {
    downloadFrame.src = examplePdfEndpoint
  })

  // Handle chat submit button click
  chatSubmit.addEventListener("click", handleChatSubmit)

  function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
  }

  function highlight() {
    dropArea.classList.add("drag-over")
  }

  function unhighlight() {
    dropArea.classList.remove("drag-over")
  }

  function handleDrop(e) {
    const dt = e.dataTransfer
    const files = dt.files

    if (files.length > 0) {
      validateAndProcessFile(files[0])
    }
  }

  function handleFileSelect(e) {
    const files = e.target.files

    if (files.length > 0) {
      validateAndProcessFile(files[0])
    }
  }

  function validateAndProcessFile(file) {
    // Reset UI elements
    errorMessage.classList.add("hidden")
    errorMessage.textContent = ""
    statusMessage.classList.add("hidden")
    statusMessage.textContent = ""

    // Reset button to upload state if it was in download state
    if (actionBtn.classList.contains("download-btn")) {
      actionBtn.classList.remove("download-btn")
      actionBtn.classList.add("upload-btn")
      actionBtn.textContent = "Upload File"
    }

    // Check file type
    const lowerCaseName = file.name.toLowerCase()

    if (lowerCaseName.endsWith(".zip") || lowerCaseName.endsWith(".pdf")) {
      // Store the selected file
      selectedFile = file

      // Determine file type
      fileType = lowerCaseName.endsWith(".zip") ? "zip" : "pdf"

      // Update file icon based on type
      if (fileType === "zip") {
        fileIcon.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 12v9a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-9"/>
            <path d="M5 12V5a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v7"/>
            <path d="M12 12v9"/>
            <path d="M12 12 8 8"/>
            <path d="m12 12 4-4"/>
          </svg>
        `
      } else {
        fileIcon.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        `
      }

      // Update file type badge
      fileTypeBadge.textContent = fileType.toUpperCase()
      fileTypeBadge.className = `file-type-badge ${fileType}`

      // Display file info
      fileName.textContent = file.name
      fileInfo.classList.remove("hidden")

      // Enable upload button
      actionBtn.disabled = false
    } else {
      // Show error for unsupported files
      errorMessage.textContent = "Please select a .zip or .pdf file"
      errorMessage.classList.remove("hidden")
      fileInfo.classList.add("hidden")
      actionBtn.disabled = true
      selectedFile = null
      fileType = null
    }
  }

  async function handleActionButtonClick() {
    // If button is in download state
    if (actionBtn.classList.contains("download-btn")) {
      // Trigger download of processed file
      initiateDownload(processedFileId)

      // Reset UI after download is initiated
      resetUI()
      return
    }

    // If button is in upload state
    if (selectedFile) {
      try {
        // Show loading state
        actionBtn.disabled = true
        actionBtn.textContent = "Processing..."
        loader.style.display = "block"

        // Create form data
        const formData = new FormData()

        // Add file to form data with the appropriate name
        formData.append(fileType === "zip" ? "file" : "file", selectedFile)

        // Add selected item IDs to form data for ZIP files
        if (fileType === "zip") {
          formData.append("selectedItemIds", JSON.stringify(selectedItemIds))
        }

        // Add prompt for PDF files
        if (fileType === "pdf") {
          formData.append("prompt", "Redact sensitive information")
        }

        // Determine which endpoint to use based on file type
        const endpoint = fileType === "zip" ? uploadEndpoint : redactEndpoint

        // Send the file to the server
        const response = await fetch(endpoint, {
          method: "POST",
          body: formData,
        })

        // Hide loader
        loader.style.display = "none"

        if (response.ok) {
          const data = await response.json()

          // Store the processed file ID
          processedFileId = data.processedFileId

          // Show success message
          statusMessage.textContent = data.message || "File processed successfully"
          statusMessage.classList.remove("hidden")

          // Change button to download state
          actionBtn.classList.remove("upload-btn")
          actionBtn.classList.add("download-btn")
          actionBtn.textContent = "Download Processed File"
          actionBtn.disabled = false
        } else {
          const errorData = await response.json()
          throw new Error(errorData.detail || "Upload failed")
        }
      } catch (error) {
        // Show error message
        errorMessage.textContent = error.message
        errorMessage.classList.remove("hidden")

        // Reset button
        actionBtn.textContent = "Upload File"
        actionBtn.disabled = false
      }
    }
  }

  function initiateDownload(fileId) {
    // Use the hidden iframe to trigger download without navigating away
    downloadFrame.src = `${downloadEndpoint}/${fileId}`

    // Show a temporary message
    statusMessage.textContent = "Download started..."

    // Disable the button temporarily during download
    actionBtn.disabled = true

    // After a short delay, reset the UI to allow for a new upload
    setTimeout(() => {
      resetUI()
    }, 1500)
  }

  function resetUI() {
    // Reset button to upload state
    actionBtn.classList.remove("download-btn")
    actionBtn.classList.add("upload-btn")
    actionBtn.textContent = "Upload File"
    actionBtn.disabled = true

    // Clear file input
    fileInput.value = ""
    selectedFile = null
    fileType = null

    // Hide file info
    fileInfo.classList.add("hidden")

    // Update status message
    statusMessage.textContent = "Ready for next file"

    // Clear processed file ID
    processedFileId = null

    // Don't reset selected items - keep them for the next upload
  }

  // Handle chat submission
  async function handleChatSubmit() {
    const message = chatInput.value.trim()

    if (!message) return

    try {
      // Show loading state
      chatSubmit.disabled = true
      chatSubmit.textContent = "Thinking..."
      chatLoader.style.display = "block"

      // Send the message to the server
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      })

      if (response.ok) {
        const data = await response.json()

        // Update the selected items based on the response
        if (data.selectedItemIds && Array.isArray(data.selectedItemIds)) {
          selectedItemIds = data.selectedItemIds
          updateSelectedItems()
          renderMultiselectOptions()
        }

        // Show success message
        chatStatusText.textContent = data.message || "Message processed successfully"
        chatStatusText.classList.remove("hidden")

        // Clear the chat input
        chatInput.value = ""
      } else {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Failed to process message")
      }
    } catch (error) {
      // Show error message
      chatStatusText.textContent = error.message
      chatStatusText.classList.remove("hidden")
    } finally {
      // Re-enable the submit button and hide loader
      chatSubmit.disabled = false
      chatSubmit.textContent = "Submit"
      chatLoader.style.display = "none"

      // Hide status message after a delay
      setTimeout(() => {
        chatStatusText.classList.add("hidden")
      }, 3000)
    }
  }

  // Initialize the multiselect dropdown
  initMultiselect()
})
