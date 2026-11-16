const recordedModeBtn = document.getElementById("recordedModeBtn");
const cameraModeBtn = document.getElementById("cameraModeBtn");

const recordedPanel = document.getElementById("recordedPanel");
const cameraPanel = document.getElementById("cameraPanel");

const modeTitle = document.getElementById("modeTitle");
const modeDescription = document.getElementById("modeDescription");

const videoInput = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const processVideoBtn = document.getElementById("processVideoBtn");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");

const processedVideo = document.getElementById("processedVideo");
const cameraPreview = document.getElementById("cameraPreview");
const processedCamera = document.getElementById("processedCamera");

const videoPlaceholder = document.getElementById("videoPlaceholder");
const processingStatus = document.getElementById("processingStatus");


// ============================================================
// CAMERA VARIABLES
// ============================================================

let cameraStream = null;
let cameraRunning = false;


// ============================================================
// RECORDED VIDEO MODE
// ============================================================

recordedModeBtn.addEventListener("click", () => {

    recordedModeBtn.classList.add("active");
    cameraModeBtn.classList.remove("active");

    recordedPanel.classList.remove("hidden");
    cameraPanel.classList.add("hidden");

    modeTitle.textContent = "Recorded Video Analysis";

    modeDescription.textContent =
        "Upload a video and analyze movement trajectories.";

    stopCamera();

});


// ============================================================
// CAMERA MODE
// ============================================================

cameraModeBtn.addEventListener("click", () => {

    cameraModeBtn.classList.add("active");
    recordedModeBtn.classList.remove("active");

    cameraPanel.classList.remove("hidden");
    recordedPanel.classList.add("hidden");

    modeTitle.textContent = "Live Camera Analysis";

    modeDescription.textContent =
        "Use your camera for real-time movement analysis.";

});


// ============================================================
// FILE SELECTION
// ============================================================

videoInput.addEventListener("change", () => {

    if (videoInput.files.length > 0) {

        fileName.textContent =
            videoInput.files[0].name;

    } else {

        fileName.textContent =
            "No video selected";

    }

});


// ============================================================
// RECORDED VIDEO PROCESSING
// ============================================================

processVideoBtn.addEventListener(
    "click",
    async () => {

        if (videoInput.files.length === 0) {

            alert("Please select a video first.");

            return;
        }


        const file = videoInput.files[0];

        const formData = new FormData();

        formData.append(
            "file",
            file
        );


        processedVideo.hidden = false;

        processedCamera.hidden = true;

        cameraPreview.hidden = true;

        videoPlaceholder.hidden = true;

        processingStatus.textContent =
            "Processing...";


        try {

            const response = await fetch(
                "/process-video",
                {
                    method: "POST",
                    body: formData
                }
            );


            if (!response.ok) {

                throw new Error(
                    "Video processing failed."
                );

            }


            processingStatus.textContent =
                "Processing";


            await readFrameStream(
                response.body
            );


            processingStatus.textContent =
                "Completed";


        } catch (error) {

            console.error(error);

            processingStatus.textContent =
                "Error";

            alert(
                "Unable to process the video."
            );

        }

    }
);


// ============================================================
// READ PROCESSED VIDEO STREAM
// ============================================================

async function readFrameStream(stream) {

    const reader = stream.getReader();

    let buffer = new Uint8Array(0);


    while (true) {

        const {
            value,
            done
        } = await reader.read();


        if (done) {

            break;

        }


        if (value) {

            buffer = combineBuffers(
                buffer,
                value
            );

        }


        while (true) {

            const start =
                findJPEGStart(buffer);

            const end =
                findJPEGEnd(
                    buffer,
                    start
                );


            if (
                start === -1 ||
                end === -1
            ) {

                break;

            }


            const jpegFrame =
                buffer.slice(
                    start,
                    end + 2
                );


            displayProcessedVideoFrame(
                jpegFrame
            );


            buffer =
                buffer.slice(
                    end + 2
                );

        }

    }

}


// ============================================================
// DISPLAY RECORDED VIDEO FRAME
// ============================================================

function displayProcessedVideoFrame(
    jpegFrame
) {

    const blob = new Blob(
        [jpegFrame],
        {
            type: "image/jpeg"
        }
    );


    const imageUrl =
        URL.createObjectURL(blob);


    processedVideo.src =
        imageUrl;


    processedVideo.onload = () => {

        URL.revokeObjectURL(
            imageUrl
        );

    };

}


// ============================================================
// START CAMERA
// ============================================================

startCameraBtn.addEventListener(
    "click",
    startCamera
);


async function startCamera() {

    try {

        processingStatus.textContent =
            "Starting Camera...";


        cameraStream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: true,
                    audio: false
                }
            );


        cameraPreview.srcObject =
            cameraStream;


        cameraPreview.hidden = false;

        processedCamera.hidden = false;

        processedVideo.hidden = true;

        videoPlaceholder.hidden = true;


        startCameraBtn.disabled = true;

        stopCameraBtn.disabled = false;


        cameraRunning = true;


        processingStatus.textContent =
            "Camera Active";


        processCameraFrames();


    } catch (error) {

        console.error(error);

        processingStatus.textContent =
            "Camera Error";


        alert(
            "Unable to access the camera. Please allow camera permission."
        );

    }

}


// ============================================================
// STOP CAMERA
// ============================================================

stopCameraBtn.addEventListener(
    "click",
    stopCamera
);


function stopCamera() {

    cameraRunning = false;


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;

    }


    cameraPreview.srcObject =
        null;


    cameraPreview.hidden = true;

    processedCamera.hidden = true;


    startCameraBtn.disabled = false;

    stopCameraBtn.disabled = true;


    processingStatus.textContent =
        "Camera Stopped";

}


// ============================================================
// PROCESS CAMERA FRAMES
// ============================================================

async function processCameraFrames() {

    if (!cameraRunning) {

        return;

    }


    const canvas =
        document.createElement("canvas");

    const context =
        canvas.getContext("2d");


    while (cameraRunning) {

        if (
            cameraPreview.readyState <
            2
        ) {

            await sleep(100);

            continue;

        }


        canvas.width =
            cameraPreview.videoWidth;

        canvas.height =
            cameraPreview.videoHeight;


        context.drawImage(
            cameraPreview,
            0,
            0,
            canvas.width,
            canvas.height
        );


        const blob =
            await canvasToBlob(
                canvas
            );


        if (!blob) {

            await sleep(100);

            continue;

        }


        try {

            const response =
                await fetch(
                    "/process-frame",
                    {
                        method: "POST",
                        body: blob
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Frame processing failed."
                );

            }


            const processedBlob =
                await response.blob();


            const imageUrl =
                URL.createObjectURL(
                    processedBlob
                );


            processedCamera.src =
                imageUrl;


            processedCamera.onload =
                () => {

                    URL.revokeObjectURL(
                        imageUrl
                    );

                };


            processingStatus.textContent =
                "Live Processing";


        } catch (error) {

            console.error(error);

            processingStatus.textContent =
                "Processing Error";

        }

    }

}


// ============================================================
// CANVAS → JPEG
// ============================================================

function canvasToBlob(canvas) {

    return new Promise(
        resolve => {

            canvas.toBlob(
                blob => {

                    resolve(blob);

                },
                "image/jpeg",
                0.8
            );

        }
    );

}


// ============================================================
// BUFFER FUNCTIONS
// ============================================================

function combineBuffers(
    first,
    second
) {

    const combined =
        new Uint8Array(
            first.length +
            second.length
        );


    combined.set(
        first,
        0
    );


    combined.set(
        second,
        first.length
    );


    return combined;

}


function findJPEGStart(
    buffer
) {

    for (
        let i = 0;
        i < buffer.length - 1;
        i++
    ) {

        if (
            buffer[i] === 0xFF &&
            buffer[i + 1] === 0xD8
        ) {

            return i;

        }

    }


    return -1;

}


function findJPEGEnd(
    buffer,
    start
) {

    if (start === -1) {

        return -1;

    }


    for (
        let i = start + 2;
        i < buffer.length - 1;
        i++
    ) {

        if (
            buffer[i] === 0xFF &&
            buffer[i + 1] === 0xD9
        ) {

            return i;

        }

    }


    return -1;

}


// ============================================================
// SLEEP
// ============================================================

function sleep(
    milliseconds
) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );

}