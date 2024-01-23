mod health;
/// Text Generation Inference Webserver
mod infer;
mod queue;
pub mod server;
mod validation;

use infer::{Infer, InferError, InferStreamResponse};
use queue::{Entry, Queue};
use serde::{Deserialize, Serialize};
use tokio::sync::OwnedSemaphorePermit;
use tokio_stream::wrappers::UnboundedReceiverStream;
use utoipa::ToSchema;
use validation::Validation;

/// Type alias for generation responses
pub(crate) type GenerateStreamResponse = (
    OwnedSemaphorePermit,
    u32, // input_length
    UnboundedReceiverStream<Result<InferStreamResponse, InferError>>,
);

/// Hub type
#[derive(Clone, Debug, Deserialize)]
pub struct HubModelInfo {
    #[serde(rename(deserialize = "id"))]
    pub model_id: String,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
}

#[derive(Clone, Deserialize, Default)]
pub struct HubTokenizerConfig {
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

impl HubTokenizerConfig {
    pub fn from_file(filename: &str) -> Self {
        let content = std::fs::read_to_string(filename).unwrap();
        serde_json::from_str(&content).unwrap_or_default()
    }
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "bigscience/blomm-560m")]
    pub model_id: String,
    #[schema(nullable = true, example = "e985a63cdc139290c5f700ff1929f0b5942cced2")]
    pub model_sha: Option<String>,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,
    #[schema(example = "cuda")]
    pub model_device_type: String,
    #[schema(nullable = true, example = "text-generation")]
    pub model_pipeline_tag: Option<String>,
    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "2")]
    pub max_best_of: usize,
    #[schema(example = "4")]
    pub max_stop_sequences: usize,
    #[schema(example = "1024")]
    pub max_input_length: usize,
    #[schema(example = "2048")]
    pub max_total_tokens: usize,
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "20")]
    pub max_waiting_tokens: usize,
    #[schema(example = "2")]
    pub validation_workers: usize,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateParameters {
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 1)]
    pub best_of: Option<usize>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 0.5
    )]
    pub temperature: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 1.03
    )]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 10)]
    pub top_k: Option<i32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub top_p: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub typical_p: Option<f32>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    #[schema(nullable = true, default = "100", example = "20")]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = false)]
    pub return_full_text: Option<bool>,
    #[serde(default)]
    #[schema(inline, max_items = 4, example = json ! (["photographer"]))]
    pub stop: Vec<String>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub truncate: Option<usize>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub watermark: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub details: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub decoder_input_details: bool,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0,
        nullable = true,
        default = "null",
        example = "null"
    )]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 5)]
    pub top_n_tokens: Option<u32>,
}

fn default_max_new_tokens() -> Option<u32> {
    Some(100)
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        best_of: None,
        temperature: None,
        repetition_penalty: None,
        top_k: None,
        top_p: None,
        typical_p: None,
        do_sample: true,
        max_new_tokens: default_max_new_tokens(),
        return_full_text: None,
        stop: Vec::new(),
        truncate: None,
        watermark: false,
        details: false,
        decoder_input_details: false,
        seed: None,
        top_n_tokens: None,
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionComplete>,
    pub usage: Usage,
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct ChatCompletionComplete {
    pub index: u32,
    pub message: Message,
    pub logprobs: Option<Vec<f32>>,
    pub finish_reason: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl ChatCompletion {
    pub(crate) fn new(
        model: String,
        system_fingerprint: String,
        output: String,
        created: u64,
        details: Details,
        return_logprobs: bool,
    ) -> Self {
        Self {
            id: String::new(),
            object: "text_completion".into(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionComplete {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: output,
                },
                logprobs: return_logprobs
                    .then(|| details.tokens.iter().map(|t| t.logprob).collect()),
                finish_reason: details.finish_reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens: details.prefill.len() as u32,
                completion_tokens: details.generated_tokens,
                total_tokens: details.prefill.len() as u32 + details.generated_tokens,
            },
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct ChatCompletionChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub logprobs: Option<f32>,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChatCompletionDelta {
    pub role: String,
    pub content: String,
}

impl ChatCompletionChunk {
    pub(crate) fn new(
        model: String,
        system_fingerprint: String,
        delta: String,
        created: u64,
        index: u32,
        logprobs: Option<f32>,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id: String::new(),
            object: "text_completion".to_string(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionChoice {
                index,
                delta: ChatCompletionDelta {
                    role: "assistant".to_string(),
                    content: delta,
                },
                logprobs,
                finish_reason,
            }],
        }
    }
}

fn default_request_messages() -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: "My name is David and I".to_string(),
    }]
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
pub(crate) struct ChatRequest {
    /// UNUSED
    #[schema(example = "bigscience/blomm-560m")]
    /// ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: String, /* NOTE: UNUSED */

    /// A list of messages comprising the conversation so far.
    #[serde(default = "default_request_messages")]
    pub messages: Vec<Message>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    /// UNUSED
    /// Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens
    /// (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically,
    /// the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model,
    /// but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should
    /// result in a ban or exclusive selection of the relevant token.
    #[serde(default)]
    pub logit_bias: Option<Vec<f32>>,

    /// Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each
    /// output token returned in the content of message.
    #[serde(default)]
    pub logprobs: Option<bool>,

    /// UNUSED
    /// An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with
    /// an associated log probability. logprobs must be set to true if this parameter is used.
    #[serde(default)]
    pub top_logprobs: Option<u32>,

    /// The maximum number of tokens that can be generated in the chat completion.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// UNUSED
    /// How many chat completion choices to generate for each input message. Note that you will be charged based on the
    /// number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
    #[serde(default)]
    pub n: Option<u32>,

    /// UNUSED
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics
    #[serde(default)]
    pub presence_penalty: Option<f32>,

    #[serde(default = "bool::default")]
    pub stream: bool,

    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ChatTemplateInputs<'a> {
    messages: Vec<Message>,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
pub(crate) struct Message {
    #[schema(example = "user")]
    pub role: String,
    #[schema(example = "My name is David and I")]
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct CompatGenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
    #[serde(default)]
    #[schema(default = "false")]
    pub stream: bool,
}

impl From<CompatGenerateRequest> for GenerateRequest {
    fn from(req: CompatGenerateRequest) -> Self {
        Self {
            inputs: req.inputs,
            parameters: req.parameters,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PrefillToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Token {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
    #[schema(example = "false")]
    special: bool,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all(serialize = "snake_case"))]
pub(crate) enum FinishReason {
    #[schema(rename = "length")]
    Length,
    #[serde(rename = "eos_token")]
    #[schema(rename = "eos_token")]
    EndOfSequenceToken,
    #[schema(rename = "stop_sequence")]
    StopSequence,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Length => write!(f, "length"),
            FinishReason::EndOfSequenceToken => write!(f, "eos_token"),
            FinishReason::StopSequence => write!(f, "stop_sequence"),
        }
    }
}

#[derive(Serialize, ToSchema)]
pub(crate) struct BestOfSequence {
    #[schema(example = "test")]
    pub generated_text: String,
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Details {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of_sequences: Option<Vec<BestOfSequence>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct GenerateResponse {
    #[schema(example = "test")]
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Details>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamDetails {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamResponse {
    pub index: u32,
    pub token: Token,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Token>,
    #[schema(nullable = true, default = "null", example = "test")]
    pub generated_text: Option<String>,
    #[schema(nullable = true, default = "null")]
    pub details: Option<StreamDetails>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: String,
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use tokenizers::Tokenizer;

    pub(crate) async fn get_tokenizer() -> Tokenizer {
        let filename = std::path::Path::new("tokenizer.json");
        if !filename.exists() {
            let content = reqwest::get("https://huggingface.co/gpt2/raw/main/tokenizer.json")
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let tmp_filename = "tokenizer.json.temp";
            let mut file = std::fs::File::create(tmp_filename).unwrap();
            file.write_all(&content).unwrap();
            // Re-check if another process has written this file maybe.
            if !filename.exists() {
                std::fs::rename(tmp_filename, filename).unwrap()
            }
        }
        Tokenizer::from_file("tokenizer.json").unwrap()
    }
}
