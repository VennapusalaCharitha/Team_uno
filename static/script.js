function speak(element) {
  let text = element.innerText || element.getAttribute("aria-label") || element.placeholder || element.value;

  if (!text || text.trim() === "") return; // Prevent speaking empty elements

  let speech = new SpeechSynthesisUtterance(text);
  speech.lang = "en-US";
  speech.rate = 1;
  speech.pitch = 1;

  window.speechSynthesis.speak(speech);
}
