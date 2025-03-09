function speak(element) {
  // Ensure sessionStorage is available
  if (typeof sessionStorage !== "undefined") {
    const isBlindValue = sessionStorage.getItem('IsBlind') ;
    // Check if IsBlind is set to "1" (blind user)
    if (isBlindValue === "1") {
      let text = element.innerText || element.getAttribute("aria-label") || element.placeholder || element.value;
      // Prevent speaking empty elements
      if (!text || text.trim() === "") return;
      
      let speech = new SpeechSynthesisUtterance(text);
      speech.lang = "en-US";
      speech.rate = 1;
      speech.pitch = 1;
  
      window.speechSynthesis.speak(speech);
    }
  }
}
