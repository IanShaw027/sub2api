//go:build unit

package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestExtractImageSize_GeminiNative 测试标准 Gemini API 的图像尺寸提取
func TestExtractImageSize_GeminiNative(t *testing.T) {
	svc := &GeminiMessagesCompatService{}

	// 测试 1K
	body := []byte(`{"generationConfig":{"imageConfig":{"imageSize":"1K"}}}`)
	require.Equal(t, "1K", svc.extractImageSize(body))

	// 测试 2K
	body = []byte(`{"generationConfig":{"imageConfig":{"imageSize":"2K"}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	// 测试 4K
	body = []byte(`{"generationConfig":{"imageConfig":{"imageSize":"4K"}}}`)
	require.Equal(t, "4K", svc.extractImageSize(body))
}

// TestExtractImageSize_GeminiNative_CaseInsensitive 测试大小写不敏感
func TestExtractImageSize_GeminiNative_CaseInsensitive(t *testing.T) {
	svc := &GeminiMessagesCompatService{}

	body := []byte(`{"generationConfig":{"imageConfig":{"imageSize":"1k"}}}`)
	require.Equal(t, "1K", svc.extractImageSize(body))

	body = []byte(`{"generationConfig":{"imageConfig":{"imageSize":"4k"}}}`)
	require.Equal(t, "4K", svc.extractImageSize(body))
}

// TestExtractImageSize_GeminiNative_Default 测试默认值
func TestExtractImageSize_GeminiNative_Default(t *testing.T) {
	svc := &GeminiMessagesCompatService{}

	// 无 generationConfig
	body := []byte(`{"contents":[]}`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	// 有 generationConfig 但无 imageConfig
	body = []byte(`{"generationConfig":{"temperature":0.7}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	// 有 imageConfig 但无 imageSize
	body = []byte(`{"generationConfig":{"imageConfig":{}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))
}

// TestExtractImageSize_GeminiNative_InvalidJSON 测试非法 JSON
func TestExtractImageSize_GeminiNative_InvalidJSON(t *testing.T) {
	svc := &GeminiMessagesCompatService{}

	body := []byte(`not valid json`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	body = []byte(`{"broken":`)
	require.Equal(t, "2K", svc.extractImageSize(body))
}

// TestExtractImageSize_GeminiNative_InvalidSize 测试无效尺寸
func TestExtractImageSize_GeminiNative_InvalidSize(t *testing.T) {
	svc := &GeminiMessagesCompatService{}

	// 不支持的尺寸
	body := []byte(`{"generationConfig":{"imageConfig":{"imageSize":"3K"}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	body = []byte(`{"generationConfig":{"imageConfig":{"imageSize":"8K"}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	body = []byte(`{"generationConfig":{"imageConfig":{"imageSize":"invalid"}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))
}

// TestExtractImageSize_GeminiNative_EmptySize 测试空 imageSize
func TestExtractImageSize_GeminiNative_EmptySize(t *testing.T) {
	svc := &GeminiMessagesCompatService{}

	body := []byte(`{"generationConfig":{"imageConfig":{"imageSize":""}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))

	// 空格
	body = []byte(`{"generationConfig":{"imageConfig":{"imageSize":"   "}}}`)
	require.Equal(t, "2K", svc.extractImageSize(body))
}

// TestIsImageGenerationModel_Consistency 测试模型识别的一致性
func TestIsImageGenerationModel_Consistency(t *testing.T) {
	// 确保 Antigravity 和标准 Gemini API 使用相同的模型识别逻辑
	testCases := []struct {
		model    string
		expected bool
	}{
		// 图像生成模型
		{"gemini-3-pro-image", true},
		{"gemini-3-pro-image-preview", true},
		{"gemini-3-pro-image-001", true},
		{"gemini-2.5-flash-image", true},
		{"gemini-2.5-flash-image-preview", true},
		{"gemini-2.5-flash-image-exp-0827", true},
		{"models/gemini-3-pro-image", true},
		{"GEMINI-3-PRO-IMAGE", true},

		// 文本生成模型
		{"gemini-2.5-pro", false},
		{"gemini-2.5-flash", false},
		{"gemini-1.5-pro", false},
		{"claude-3-opus", false},
		{"gpt-4o", false},

		// 边界情况
		{"my-gemini-3-pro-image-test", false}, // 不应误匹配自定义模型
		{"custom-gemini-2.5-flash-image-wrapper", false},
	}

	for _, tc := range testCases {
		t.Run(tc.model, func(t *testing.T) {
			result := isImageGenerationModel(tc.model)
			require.Equal(t, tc.expected, result, "Model: %s", tc.model)
		})
	}
}
