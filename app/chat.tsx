import React, { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import {
  TextInput,
  Button,
  ScrollView,
  Text,
  View,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Image,
} from "react-native";
import * as Speech from "expo-speech";
import * as ImagePicker from "expo-image-picker";


const GROQ_API_KEY = "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";
const MODEL_API_URL = "http://10.16.17.198:5005/predict"; 

interface Message {
  id?: string;
  role: "user" | "bot";
  content: string;
  image?: string;
  grad?: string;  
  lime?: string; 
}


const Chat = () => {
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [message, setMessage] = useState("");
  const [userId, setUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserId = async () => {
      const { data, error } = await supabase.auth.getUser();
      if (error) console.error("Error fetching user ID:", error);
      else setUserId(data?.user?.id || null);
    };
    fetchUserId();
  }, []);

  useEffect(() => {
    const fetchChatHistory = async () => {
      if (!userId) return;
      const { data, error } = await supabase
        .from("chats")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: true });

      if (error) {
        console.error("Error fetching chat history:", error);
      } else {
        setChatHistory(
          data.map((msg) => ({
            id: msg.id,
            role: msg.message.startsWith("User:") ? "user" : "bot",
            content: msg.message.replace(/^User: |^Bot: /, ""),
          }))
        );
      }
    };
    fetchChatHistory();
  }, [userId]);

  const pickImage = async (fromCamera: boolean) => {
    let result: ImagePicker.ImagePickerResult;

    if (fromCamera) {
      result = await ImagePicker.launchCameraAsync({ allowsEditing: true, base64: true });
    } else {
      result = await ImagePicker.launchImageLibraryAsync({ allowsEditing: true, base64: true });
    }

    if (!result.canceled && result.assets?.[0]?.uri) {
      setImage(result.assets[0].uri);
      preprocessAndPredict(result.assets[0].uri);
    }
  };
 
 
  const sendMessage = async () => {
    if (!message.trim()) return;
  
    setLoading(true);
    const userMessage: Message = { role: "user", content: message };
    setChatHistory((prev) => [...prev, userMessage]);
  
    try {
      await supabase.from("chats").insert([{ user_id: userId, message: `User: ${message}` }]);
  
      const response = await fetch("http://10.16.17.198:5005/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: message,
          chat_history: chatHistory.map((msg) => [msg.role === "user" ? msg.content : "", msg.role === "bot" ? msg.content : ""]),
        }),
      });
  
      const data = await response.json();
  
      const bookAnswer = data?.book_answer || "No answer from book.";
      const webAnswer = data?.web_answer || "No answer from web.";
  
      const combinedResponse = `From Book:\n${bookAnswer}\n\nFrom Web:\n${webAnswer}`;
      const botMessage: Message = { role: "bot", content: combinedResponse };
  
      setChatHistory((prev) => [...prev, botMessage]);
      Speech.speak(combinedResponse);
  
      await supabase.from("chats").insert([{ user_id: userId, message: `Bot: ${combinedResponse}` }]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setLoading(false);
      setMessage("");
    }
  };
  const base64ToBlob = (base64: string, contentType = 'image/jpeg') => {
    const byteCharacters = atob(base64);
    const byteArrays = [];
  
    for (let i = 0; i < byteCharacters.length; i += 512) {
      const slice = byteCharacters.slice(i, i + 512);
      const byteNumbers = new Array(slice.length);
  
      for (let j = 0; j < slice.length; j++) {
        byteNumbers[j] = slice.charCodeAt(j);
      }
  
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
  
    return new Blob(byteArrays, { type: contentType });
  };
  
const uploadXAIImage = async (base64Image: string, type: "gradcam" | "lime"): Promise<string | null> => {
    try {
      const fileName = `${type}-${Date.now()}.jpg`;
      const blob = base64ToBlob(base64Image, "image/jpeg");
  
      
      const { error } = await supabase.storage
        .from("chat-images")
        .upload(`xai/${fileName}`, blob, { upsert: true });
  
      if (error) {
        console.error(`❌ Failed to upload ${type} image:`, error.message);
        return null;
      }
  
      const { data: publicUrlData } = supabase.storage
        .from("chat-images")
        .getPublicUrl(`xai/${fileName}`);
  
      const publicUrl = publicUrlData?.publicUrl ?? null;
      if (!publicUrl) {
        console.error(`❌ Could not retrieve public URL for ${type} image`);
        return null;
      }
  
      console.log(`✅ Uploaded ${type} image to Supabase`);
      return publicUrl;
  
    } catch (err) {
      console.error(`🔥 Error uploading ${type} image:`, err);
      return null;
    }
  };
  

  const preprocessAndPredict = async (imageUri: string) => {
    try {
      setLoading(true);
      console.log("🟠 Starting preprocessAndPredict...");
  
      // 🔹 Resize image
      const resizedImage = await manipulateAsync(
        imageUri,
        [{ resize: { width: 224, height: 224 } }],
        { format: SaveFormat.JPEG, base64: true }
      );
  
      if (!resizedImage.base64) {
        throw new Error("❌ Failed to convert image to Base64");
      }
      console.log("✅ Image resized & base64 encoded");
  
      // 🔁 Send to /upload endpoint for preprocessing
      const uploadResponse = await fetch("http://10.16.17.198:5005/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: resizedImage.base64 }),
      });
  
      const uploadText = await uploadResponse.text();
      let backendData;
      try {
        backendData = JSON.parse(uploadText);
      } catch (parseError) {
        console.error("❌ Could not parse /upload response as JSON.");
        console.error("🔎 Response Text:", uploadText);
        throw parseError;
      }
  
      if (backendData.error) throw new Error(`❌ Backend Error: ${backendData.error}`);
      console.log("✅ Received preprocessed image from /upload");
  
      // 🔁 Send preprocessed image to model
      const modelResponse = await fetch(MODEL_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: backendData.preprocessed_image }),
      });
  
      const modelText = await modelResponse.text();
      let modelData;
      try {
        modelData = JSON.parse(modelText);
      } catch (jsonError) {
        console.error("❌ Could not parse model response as JSON.");
        console.error("🔎 Model Response Text:", modelText);
        throw jsonError;
      }
  
      if (!modelResponse.ok) throw new Error(modelData.error || "Prediction failed");
      console.log("✅ Model Prediction:", modelData.prediction);
  
      // 🖼️ Upload preprocessed image to Supabase
      const fileName = `preprocessed-${Date.now()}.jpg`;

      const dataUrl = `data:image/jpeg;base64,${backendData.preprocessed_image}`;
      const blob = await (await fetch(dataUrl)).blob();
     
      const { error: uploadError } = await supabase.storage
        .from("chat-images")
        .upload(`processed/${fileName}`, blob, { upsert: true });
  
      if (uploadError) {
        console.error("❌ Supabase image upload failed:", uploadError.message);
      }
  
      const { data: publicUrlData } = supabase.storage
        .from("chat-images")
        .getPublicUrl(`processed/${fileName}`);
      const imageUrl = publicUrlData?.publicUrl ?? null;
  
      // 🔁 Fetch Grad-CAM and LIME from /xai
      const xaiResponse = await fetch("http://10.16.17.198:5005/xai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
  
      const xaiText = await xaiResponse.text();
      let xaiData;
  
      try {
        xaiData = JSON.parse(xaiText);
      } catch (parseError) {
        console.error("❌ Could not parse /xai response as JSON.");
        console.error("🔎 Response Text:", xaiText);
        throw parseError;
      }
  
      if (xaiData.error) {
        throw new Error(`❌ XAI Error: ${xaiData.error}`);
      }
  
      console.log("✅ Received XAI images from /xai");
  
      // Upload Grad-CAM & LIME to Supabase
      const gradcamUrl = await uploadXAIImage(xaiData.gradcam, "gradcam");
      const limeUrl = await uploadXAIImage(xaiData.lime, "lime");
  
      if (!gradcamUrl || !limeUrl) {
        throw new Error("❌ Upload to Supabase failed for Grad-CAM or LIME");
      }
  
      const botMessage: Message = {
        role: "bot",
        content: `Prediction: ${modelData.prediction}`,
        image: imageUrl ?? undefined,
        grad: gradcamUrl ?? undefined,  // This will work now
        lime: limeUrl ?? undefined,    // This will work now
      };
      
      setChatHistory((prev) => [...prev, botMessage]);
      Speech.speak(botMessage.content);
  
      // 🧾 Save to Supabase database
      await supabase.from("chats").insert([
        {
          user_id: userId,
          message: `Bot: Prediction: ${modelData.prediction}`,
          image_url: imageUrl,
          gradcam_url: gradcamUrl,
          lime_url: limeUrl,
        },
      ]);
  
      console.log("✅ Chat + images inserted into Supabase");
  
    } catch (error: unknown) {
      if (error instanceof Error) {
        console.error("🔥 Image Processing Error:", error.message);
      } else {
        console.error("🔥 Unknown error:", error);
      }
    } finally {
      setLoading(false);
    }
  };
  
  



  
  
  return (
    <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={{ flex: 1, padding: 16 }}>
      <ScrollView style={{ flex: 1 }} contentContainerStyle={{ paddingBottom: 20 }}>
        {chatHistory.map((msg, index) => (
          <View
            key={index}
            style={{
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              backgroundColor: msg.role === "user" ? "#007AFF" : "#E5E5EA",
              padding: 10,
              borderRadius: 8,
              marginVertical: 4,
              maxWidth: "75%",
            }}
          >
            {/* Show preprocessed image if available */}
            {msg.image && (
              <View style={{ marginBottom: 5 }}>
                <Text style={{ fontSize: 12, color: msg.role === "user" ? "#eee" : "#555", marginBottom: 4 }}>
                  Preprocessed Image:
                </Text>
                <Image
                  source={{ uri: msg.image }}
                  style={{ width: 200, height: 200, borderRadius: 8 }}
                  resizeMode="cover"
                />
              </View>
            )}



{msg.grad && (
              <View style={{ marginBottom: 5 }}>
                <Text style={{ fontSize: 12, color: msg.role === "user" ? "#eee" : "#555", marginBottom: 4 }}>
                  GradCam:
                </Text>
                <Image
                  source={{ uri: msg.grad}}
                  style={{ width: 200, height: 200, borderRadius: 8 }}
                  resizeMode="cover"
                />
              </View>
            )}



{msg.lime && (
              <View style={{ marginBottom: 5 }}>
                <Text style={{ fontSize: 12, color: msg.role === "user" ? "#eee" : "#555", marginBottom: 4 }}>
                  Lime:
                </Text>
                <Image
                  source={{ uri: msg.lime}}
                  style={{ width: 200, height: 200, borderRadius: 8 }}
                  resizeMode="cover"
                />
              </View>
            )}


  
            <Text style={{ color: msg.role === "user" ? "white" : "black" }}>{msg.content}</Text>
          </View>
        ))}
        {loading && <ActivityIndicator size="small" color="#007AFF" style={{ marginTop: 10 }} />}
      </ScrollView>
  
      <View style={{ flexDirection: "row", alignItems: "center", marginTop: 10 }}>
        <TextInput
          style={{ flex: 1, borderWidth: 1, borderColor: "#ddd", borderRadius: 8, padding: 10 }}
          value={message}
          onChangeText={setMessage}
          placeholder="Type a message..."
        />
        <Button title="Send" onPress={sendMessage} disabled={loading} />
      </View>
  
      <View style={{ flexDirection: "row", justifyContent: "space-evenly", marginTop: 10 }}>
        <Button title="Upload Image" onPress={() => pickImage(false)} />
        <Button title="Take Photo" onPress={() => pickImage(true)} />
      </View>
    </KeyboardAvoidingView>
  );
  
}
export default Chat;
