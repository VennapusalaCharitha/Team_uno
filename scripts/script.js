// Function to simulate joining a meeting
function joinMeeting() {
  const meetingID = document.getElementById('meeting-id').value;
  const passcode = document.getElementById('passcode').value;

  if (meetingID && passcode) {
    document.querySelector('.home-screen').hidden = true;
    document.querySelector('.meeting-screen').hidden = false;
    alert(`Joined Meeting ${meetingID}`);
  } else {
    alert("Please enter a valid Meeting ID and Passcode.");
  }
}

// Function to simulate toggling mute
function toggleMute() {
  const muteButton = document.getElementById('mute-button');
  if (muteButton.innerText === "Mute") {
    muteButton.innerText = "Unmute";
    alert("Muted");
  } else {
    muteButton.innerText = "Mute";
    alert("Unmuted");
  }
}

// Function to simulate toggling video
function toggleVideo() {
  const videoButton = document.getElementById('video-button');
  if (videoButton.innerText === "Turn off Video") {
    videoButton.innerText = "Turn on Video";
    alert("Video turned off");
  } else {
    videoButton.innerText = "Turn off Video";
    alert("Video turned on");
  }
}

// Function to simulate leaving the meeting
function leaveMeeting() {
  const confirmation = confirm("Are you sure you want to leave the meeting?");
  if (confirmation) {
    document.querySelector('.meeting-screen').hidden = true;
    document.querySelector('.home-screen').hidden = false;
    alert("Left the meeting");
  }
}

// Voice Commands (optional for better accessibility)
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.onstart = function () {
  console.log("Voice recognition started");
};

recognition.onresult = function (event) {
  const command = event.results[0][0].transcript.toLowerCase();
  if (command.includes("join")) {
    joinMeeting();
  } else if (command.includes("mute")) {
    toggleMute();
  } else if (command.includes("video")) {
    toggleVideo();
  } else if (command.includes("leave")) {
    leaveMeeting();
  }
};

// Start voice recognition for commands
function startVoiceRecognition() {
  recognition.start();
}

// Enable voice command feature on the page
window.onload = function() {
  const voiceButton = document.createElement('button');
  voiceButton.innerText = "Start Voice Command";
  voiceButton.onclick = startVoiceRecognition;
  document.body.appendChild(voiceButton);
};
