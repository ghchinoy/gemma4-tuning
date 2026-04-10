import Foundation

struct LogEntry: Identifiable, Decodable {
    var id: String { return "\(type)-\(iter ?? 0)-\(status ?? "")" }
    let type: String
    let iter: Int?
    let loss: Double?
    
    // Optional fields because train and val emit different metrics
    let learning_rate: Double?
    let it_sec: Double?
    let tokens_sec: Double?
    let trained_tokens: Int?
    let peak_mem_gb: Double?
    let val_took_s: Double?
    
    // Status fields
    let status: String?
    let timestamp: String?
    
    // Config fields
    let model: String?
    let data: String?
    let total_iters: Int?
    let batch_size: Int?
    
    enum CodingKeys: String, CodingKey {
        case type, iter, loss
        case learning_rate, it_sec, tokens_sec, trained_tokens, peak_mem_gb
        case val_took_s
        case status, timestamp
        case model, data, total_iters, batch_size
    }
}
