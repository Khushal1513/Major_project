const CHAT_API =
"http://localhost:8000/ask";

const UPLOAD_API =
"http://localhost:8000/upload";

let documentId = null;

async function sendMessage() {

    const input =
    document.getElementById("questionInput");

    const chatBox =
    document.getElementById("chatBox");

    const question =
    input.value.trim();

    if(question === "") return;

    chatBox.innerHTML +=
    `<div class="user-message">${question}</div>`;

    input.value="";

    const loading =
    document.createElement("div");

    loading.className="bot-message";
    loading.innerText="Thinking...";

    chatBox.appendChild(loading);

    try{

        const response =
        await fetch(CHAT_API,{

            method:"POST",

            headers:{
                "Content-Type":"application/json"
            },

            body:JSON.stringify({
                query:question,
                top_k:5
            })
        });

        const data =
        await response.json();

        loading.innerText =
        data.answer;

    }
    catch(error){

        loading.innerText =
        "Server Error";

        console.log(error);
    }

    chatBox.scrollTop =
    chatBox.scrollHeight;
}

async function uploadDocument(){

    const fileInput =
    document.getElementById("fileInput");

    const file =
    fileInput.files[0];

    if(!file){
        alert("Select a file");
        return;
    }

    const formData =
    new FormData();

    formData.append(
        "file",
        file
    );

    document.getElementById(
        "uploadStatus"
    ).innerText =
    "Uploading...";

    try{

        const response =
        await fetch(
            UPLOAD_API,
            {
                method:"POST",
                body:formData
            }
        );

        const data =
        await response.json();

        documentId =
        data.document_id;

        document.getElementById(
            "uploadStatus"
        ).innerHTML =
        "✅ Document Processed";

        alert(
            "Document uploaded successfully!"
        );

    }
    catch(error){

        document.getElementById(
            "uploadStatus"
        ).innerHTML =
        "❌ Upload Failed";

        console.log(error);
    }
}

document
.getElementById("questionInput")
.addEventListener("keypress",
function(e){

    if(e.key==="Enter"){
        sendMessage();
    }
});