// File: /web-client/api/proxy.js

export const config = {
  // Tối ưu hóa: API nhận ảnh base64 có thể lớn, cấu hình body size limit (Mặc định của Vercel Serverless là 4.5MB)
  api: {
    bodyParser: {
      sizeLimit: '4mb', 
    },
  },
};

export default async function handler(req, res) {
  // Chỉ chấp nhận POST request
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  // Khai báo biến môi trường từ hệ thống Vercel
  const azureEndpoint = process.env.AZURE_ML_ENDPOINT;
  const azureApiKey = process.env.AZURE_ML_KEY;

  if (!azureEndpoint || !azureApiKey) {
    console.error("Missing Azure ML Environment Variables");
    return res.status(500).json({ error: 'Server configuration error' });
  }

  try {
    // 1. Trích xuất Base64 từ Frontend gửi lên
    const { image } = req.body;

    if (!image) {
      return res.status(400).json({ error: 'No image data provided' });
    }

    // 2. Chuyển tiếp Request sang Azure ML
    // Lưu ý: Cấu trúc JSON payload ở đây phụ thuộc vào cách bạn viết file score.py trên Azure.
    // Dưới đây là định dạng mặc định thường dùng, hãy điều chỉnh cho khớp với model của bạn.
    const azurePayload = {
      "input_data": {
        "columns": ["image"],
        "data": [image]
      }
    };

    const startTime = Date.now();

    const response = await fetch(azureEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${azureApiKey}`
      },
      body: JSON.stringify(azurePayload)
    });

    const inferenceTime = Date.now() - startTime;

    // 3. Bắt lỗi trả về từ Azure (Timeout, 401, 500...)
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Azure ML Error (${response.status}):`, errorText);
      return res.status(response.status).json({ 
        error: 'Inference failed on Azure ML',
        details: response.status === 401 ? 'Unauthorized' : 'Gateway Timeout/Error' 
      });
    }

    // 4. Bóc tách và chuẩn hóa JSON trả về cho Frontend
    const azureData = await response.json();
    
    // Gửi kèm thời gian xử lý thực tế của Azure (Inference Time) để phục vụ cho Bảng Telemetry ngày mai
    return res.status(200).json({
      success: true,
      data: azureData,
      telemetry: {
        inference_ms: inferenceTime
      }
    });

  } catch (error) {
    // Bắt lỗi hệ thống (ví dụ: mất kết nối mạng giữa Vercel và Azure)
    console.error("Proxy execution error:", error);
    return res.status(500).json({ error: 'Internal Server Error processing request' });
  }
}