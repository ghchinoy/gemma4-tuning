import SwiftUI
import Charts

struct ContentView: View {
    @StateObject private var monitor = LogMonitor()
    @State private var filePath: String = "~/projects/gemmma/training_log.jsonl"
    @State private var selectedIteration: Int?
    
    var body: some View {
        VStack {
            HStack {
                TextField("Log file path", text: $filePath)
                    .textFieldStyle(.roundedBorder)
                Button("Monitor") {
                    monitor.startMonitoring(filePath: filePath)
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
            
            HStack {
                if let start = monitor.startTime {
                    Text("Started: \(formatDate(start))")
                }
                Spacer()
                if monitor.isTraining {
                    ProgressView().scaleEffect(0.5).padding(.trailing, 4)
                    Text("Training...")
                        .foregroundStyle(.blue)
                } else if let end = monitor.endTime {
                    Text("Completed: \(formatDate(end))")
                        .foregroundStyle(.green)
                }
            }
            .padding(.horizontal)
            .font(.subheadline)
            
            if let config = monitor.config {
                Text("Model: \(config.model ?? "Unknown") | Data: \(config.data ?? "Unknown") | Iters: \(config.total_iters ?? 0) | Batch Size: \(config.batch_size ?? 0)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }
            
            Chart {
                ForEach(monitor.trainLogs) { entry in
                    if let iter = entry.iter, let loss = entry.loss {
                        LineMark(
                            x: .value("Iteration", iter),
                            y: .value("Loss", loss)
                        )
                        .foregroundStyle(by: .value("Type", "Train Loss"))
                        .lineStyle(StrokeStyle(lineWidth: 1.5))
                        .opacity(0.8)
                    }
                }
                ForEach(monitor.valLogs) { entry in
                    if let iter = entry.iter, let loss = entry.loss {
                        LineMark(
                            x: .value("Iteration", iter),
                            y: .value("Loss", loss)
                        )
                        .foregroundStyle(by: .value("Type", "Validation Loss"))
                        .lineStyle(StrokeStyle(lineWidth: 2))
                        
                        PointMark(
                            x: .value("Iteration", iter),
                            y: .value("Loss", loss)
                        )
                        .foregroundStyle(by: .value("Type", "Validation Loss"))
                        .symbolSize(40)
                    }
                }
                
                // Hover overlay
                if let rawIter = selectedIteration {
                    let maxIter = max(monitor.trainLogs.last?.iter ?? 0, monitor.valLogs.last?.iter ?? 0)
                    if maxIter > 0 {
                        let selectedIter = max(0, min(rawIter, maxIter))
                        
                        RuleMark(x: .value("Selected Iteration", selectedIter))
                            .foregroundStyle(Color.gray.opacity(0.5))
                            .annotation(position: .top) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Iter: \(selectedIter)")
                                        .font(.caption.bold())
                                    
                                    let trainMatch = monitor.trainLogs.min(by: { abs(($0.iter ?? 0) - selectedIter) < abs(($1.iter ?? 0) - selectedIter) })
                                    let valMatch = monitor.valLogs.min(by: { abs(($0.iter ?? 0) - selectedIter) < abs(($1.iter ?? 0) - selectedIter) })
                                    
                                    if let v = valMatch, let vIter = v.iter, abs(vIter - selectedIter) <= 50, let loss = v.loss {
                                        Text("Val: \(String(format: "%.3f", loss))")
                                            .font(.caption)
                                            .foregroundStyle(.red)
                                    }
                                    if let t = trainMatch, let tIter = t.iter, abs(tIter - selectedIter) <= 50, let loss = t.loss {
                                        Text("Train: \(String(format: "%.3f", loss))")
                                            .font(.caption)
                                            .foregroundStyle(.blue)
                                    }
                                }
                                .padding(6)
                                .background(Color(NSColor.windowBackgroundColor).opacity(0.95))
                                .cornerRadius(6)
                                .shadow(radius: 2)
                            }
                    }
                }
            }
            .chartXSelection(value: $selectedIteration)
            .chartXAxisLabel("Iteration")
            .chartYAxisLabel("Loss")
            .chartForegroundStyleScale([
                "Train Loss": .blue,
                "Validation Loss": .red
            ])
            .padding()
            
            HStack {
                if let lastTrain = monitor.trainLogs.last, let loss = lastTrain.loss {
                    Text("Train Loss: **\(String(format: "%.3f", loss))**")
                    if let peakMem = lastTrain.peak_mem_gb {
                        Text("• Peak Mem: **\(String(format: "%.2f", peakMem)) GB**")
                    }
                    if let tokens = lastTrain.tokens_sec {
                        Text("• Speed: **\(String(format: "%.0f", tokens)) tok/s**")
                    }
                }
                Spacer()
                if let bestLoss = monitor.bestValLoss, let bestIter = monitor.bestValIter {
                    Text("Best Val: **\(String(format: "%.3f", bestLoss))** (Iter \(bestIter))")
                        .foregroundStyle(.green)
                }
            }
            .padding(.horizontal)
            .padding(.bottom)
            .foregroundStyle(.secondary)
        }
        .frame(minWidth: 700, minHeight: 500)
    }
    
    func formatDate(_ isoString: String) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        
        if let date = formatter.date(from: isoString) ?? ISO8601DateFormatter().date(from: isoString) {
            let outFormatter = DateFormatter()
            outFormatter.timeStyle = .medium
            outFormatter.dateStyle = .short
            return outFormatter.string(from: date)
        }
        return isoString
    }
}
