import Foundation
import Combine

class LogMonitor: ObservableObject {
    @Published var trainLogs: [LogEntry] = []
    @Published var valLogs: [LogEntry] = []
    @Published var startTime: String?
    @Published var endTime: String?
    @Published var isTraining: Bool = false
    @Published var config: LogEntry?
    @Published var bestValLoss: Double?
    @Published var bestValIter: Int?
    
    private var fileURL: URL?
    private var fileHandle: FileHandle?
    private var timer: Timer?
    
    func startMonitoring(filePath: String) {
        timer?.invalidate()
        trainLogs.removeAll()
        valLogs.removeAll()
        startTime = nil
        endTime = nil
        config = nil
        bestValLoss = nil
        bestValIter = nil
        isTraining = true
        
        let expandedPath = NSString(string: filePath).expandingTildeInPath
        let url = URL(fileURLWithPath: expandedPath)
        self.fileURL = url
        
        do {
            fileHandle = try FileHandle(forReadingFrom: url)
            
            // Read initial content
            readNewLines()
            
            // Poll for changes
            timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
                self?.readNewLines()
            }
        } catch {
            print("Error opening file: \(error)")
        }
    }
    
    private func readNewLines() {
        guard let handle = fileHandle else { return }
        let data = handle.readDataToEndOfFile()
        guard !data.isEmpty, let string = String(data: data, encoding: .utf8) else { return }
        
        let lines = string.components(separatedBy: .newlines).filter { !$0.isEmpty }
        let decoder = JSONDecoder()
        
        var newTrain: [LogEntry] = []
        var newVal: [LogEntry] = []
        
        for line in lines {
            if let lineData = line.data(using: .utf8),
               let entry = try? decoder.decode(LogEntry.self, from: lineData) {
                if entry.type == "train" {
                    newTrain.append(entry)
                } else if entry.type == "val" {
                    newVal.append(entry)
                    
                    // Track the sweet spot
                    if let loss = entry.loss, let it = entry.iter {
                        DispatchQueue.main.async {
                            if self.bestValLoss == nil || loss < self.bestValLoss! {
                                self.bestValLoss = loss
                                self.bestValIter = it
                            }
                        }
                    }
                } else if entry.type == "config" {
                    DispatchQueue.main.async { self.config = entry }
                } else if entry.type == "status" {
                    if entry.status == "started" {
                        DispatchQueue.main.async { self.startTime = entry.timestamp }
                    } else if entry.status == "completed" {
                        DispatchQueue.main.async { 
                            self.endTime = entry.timestamp 
                            self.isTraining = false
                            self.timer?.invalidate()
                        }
                    }
                }
            }
        }
        
        if !newTrain.isEmpty || !newVal.isEmpty {
            DispatchQueue.main.async {
                self.trainLogs.append(contentsOf: newTrain)
                self.valLogs.append(contentsOf: newVal)
            }
        }
    }
}
