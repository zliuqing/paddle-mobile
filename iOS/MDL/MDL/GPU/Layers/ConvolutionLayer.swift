/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/


import Foundation
import MetalPerformanceShaders

@available(iOS 10.0, *)
class ConvolutionDataSource: NSObject, MPSCNNConvolutionDataSource {
    var activation: MPSCNNNeuron?
    var weightWidth = 1
    var weightHeight = 1
    var inputChannels = 1
    var outputChannels = 1
    var stride = 1
    var weight: UnsafeMutableRawPointer
    var bias: UnsafeMutablePointer<Float>?
    
    init(weightWidth: Int, weightHeight: Int, inputChannels: Int, outputChannels: Int, weight: UnsafeMutableRawPointer, bias: UnsafeMutablePointer<Float>?, stride: Int, activation: MPSCNNNeuron?) {
        self.bias = bias
        self.weight = weight
        self.weightWidth = weightWidth
        self.weightHeight = weightHeight
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.stride = stride
        self.activation = activation
        super.init()
    }
    
    func dataType() -> MPSDataType {
        return MPSDataType.float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        let desc = MPSCNNConvolutionDescriptor.init(kernelWidth: weightWidth,
                                                    kernelHeight: weightHeight,
                                                    inputFeatureChannels: inputChannels,
                                                    outputFeatureChannels: outputChannels,
                                                    neuronFilter: activation)
        desc.strideInPixelsX = stride
        desc.strideInPixelsY = stride
        return desc
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return self.weight
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return self.bias
    }
    
    func load() -> Bool {
        return true
    }
    
    func purge() {
    }
    
    public func label() -> String?{
        return nil
    }
}

@available(iOS 10.0, *)
class ConvolutionLayer: MPSCNNLayer {
    let padding: Int
    var conv: MPSCNNConvolution?
    var activation: MPSCNNNeuron?
    var kernelSize: Int
    var stride: Int
    
    public override init(device: MTLDevice,
                         config: LayerModel) throws {
        guard config.weight.count > 0 else {
            throw NetError.modelDataError(message: "weight of \(config.name) has no elelment")
        }
        kernelSize = config.param?.kernel_size  ?? 0
        self.stride = config.param?.stride ?? 0
        self.padding = config.param?.pad ?? 0
        if config.relu{
            activation = MPSCNNNeuronReLU(device: device, a: config.reluA)
        }
        
        try super.init(device: device, config: config)
    }
    
    override var type: String{
        return LayerModel.convolutionType
    }
    
    override func initializeCompute(device: MTLDevice){
        guard weights.count > 0 && inputs.count > 0 && outputs.count > 0 else {
            fatalError("weights , inputs or outputs has no element")
        }
        
        let weight = weights[0]
        let input = inputs[0]
        let output = outputs[0]
        
        var bias: Matrix?
        if useBias {
            bias = weights[1]
        }
        guard let wData = weight.data?.pointer else {
            fatalError("weight data is nil")
        }
        
        let dataSource = ConvolutionDataSource.init(weightWidth: weight.width,
                                                    weightHeight: weight.height,
                                                    inputChannels: input.channels,
                                                    outputChannels: output.channels,
                                                    weight: wData,
                                                    bias: bias?.data?.pointer,
                                                    stride: stride, activation: activation)
        
        if #available(iOS 11.0, *) {
            conv = MPSCNNConvolution.init(device: device, weights: dataSource)
        } else {
            //            let method: Method = class_getClassMethod(MPSCNNConvolution, Selector.init(""))
            //
            //            let implementation = method_getImplementation(method)
            //
            //            typealias Function = @convention(c) (AnyObject, Selector, MTLDevice, MPSCNNConvolutionDescriptor, UnsafePointer<Float>, UnsafePointer<Float>?, MPSCNNConvolutionFlags) -> Unmanaged<MPSCNNConvolution>
            //
            //
            //            let function = unsafeBitCast(implementation, to: Function.self)
            
            /*
             public init(device: MTLDevice, convolutionDescriptor: MPSCNNConvolutionDescriptor, kernelWeights: UnsafePointer<Float>, biasTerms: UnsafePointer<Float>?, flags: MPSCNNConvolutionFlags)
             */
            
            
        }
        conv?.edgeMode = .zero
        mpscnn = conv
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer) {
        guard inputs.count > 0 && outputs.count > 0 else{
            fatalError("inputs or outputs has no element")
        }
        
        let input = inputs[0]
        let output = outputs[0]
        
        conv?.offset = MetalManager.offsetForConvolution(padding: self.padding,
                                                         sourceWidth: input.width,
                                                         sourceHeight: input.height,
                                                         destinationWidth: output.width,
                                                         destinationHeight: output.height,
                                                         kernelWidth: kernelSize,
                                                         kernelHeight: kernelSize,
                                                         strideInPixelsX: stride,
                                                         strideInPixelsY: stride)
        
        super.encode(commandBuffer: commandBuffer)
    }
}

@available(iOS 10.0, *)
class DepthwiseConvolution: Layer {
    let kernel: (Int, Int)
    let stride: (Int, Int)
    var activation: MPSCNNNeuron?
    var compute: Any!
    var pad: Int
    override init(device: MTLDevice,
                  config: LayerModel) throws {
        guard config.weight.count > 0 else {
            throw NetError.modelDataError(message: "weight of \(config.name) has no elelment")
        }
        self.kernel = (config.param?.kernel_size ?? 0, config.param?.kernel_size ?? 0)
        self.stride = (config.param?.stride ?? 0, config.param?.stride ?? 0)
        if config.relu {
            self.activation = MPSCNNNeuronReLU(device: device, a: 0)
        }
        self.pad = config.param?.pad ?? 0
        try super.init(device: device, config: config)
    }
    
    override var type: String {
        return LayerModel.depthWiseConvolutionType
    }
    
    override func initializeCompute(device: MTLDevice) {
        guard inputs.count > 0 && weights.count > 0 else {
            fatalError("inputs or weights has no element")
        }
        guard let weightData = weights[0].data?.pointer else{
            fatalError("weight data is nil")
        }
        let input = inputs[0]
        var biasTerms: UnsafeMutablePointer<Float>?
        if useBias {
            biasTerms = weights[1].data?.pointer
        }
        
        //        if #available(iOS 11.0, *) {
        //            let desc = MPSCNNDepthWiseConvolutionDescriptor.init(kernelWidth: kernel.0, kernelHeight: kernel.1,
        //                                                                 inputFeatureChannels: input.channels,
        //                                                                 outputFeatureChannels: input.channels,
        //                                                                 neuronFilter: activation)
        //            desc.strideInPixelsX = stride.0
        //            desc.strideInPixelsY = stride.1
        //
        //            compute = MPSCNNConvolution.init(device: device,
        //                                             convolutionDescriptor: desc,
        //                                             kernelWeights: weightData,
        //                                             biasTerms: biasTerms,
        //                                             flags: .none)
        //            (compute as! MPSCNNConvolution).edgeMode = .zero
        //
        //        }else{
        compute = DepthwiseConvolutionKernel(device: device,
                                             kernelWidth: kernel.0,
                                             kernelHeight: kernel.1,
                                             featureChannels: input.channels,
                                             strideInPixelsX: stride.0,
                                             strideInPixelsY: stride.1,
                                             channelMultiplier: 1,
                                             neuronFilter: activation,
                                             kernelWeights: weightData,
                                             biasTerms: biasTerms)
    }
    //    }
    
    override func encode(commandBuffer: MTLCommandBuffer) {
        let input = inputs[0]
        let output = outputs[0]
        let offset = MetalManager.offsetForConvolution(padding: self.pad,
                                                       sourceWidth: input.width,
                                                       sourceHeight: input.height,
                                                       destinationWidth: output.width,
                                                       destinationHeight: output.height,
                                                       kernelWidth: kernel.0,
                                                       kernelHeight: kernel.1,
                                                       strideInPixelsX: stride.0,
                                                       strideInPixelsY: stride.1)
        
        if let inCompute = compute as? DepthwiseConvolutionKernel {
            inCompute.offset = offset
            inCompute.encode(commandBuffer: commandBuffer,
                             sourceImage: input.image!,
                             destinationImage: output.image!)
        } else if let inCompute = compute as? MPSCNNConvolution {
            inCompute.offset = offset
            inCompute.encode(commandBuffer: commandBuffer,
                             sourceImage: input.image!,
                             destinationImage: output.image!)
        }
    }
}

@available(iOS 10.0, *)
class ReluLayer: MPSCNNLayer {
    var activation: MPSCNNNeuronReLU?
    override init(device: MTLDevice,
                  config: LayerModel) throws {
        try super.init(device: device, config: config)
    }
    
    override func initializeCompute(device: MTLDevice) {
        activation = MPSCNNNeuronReLU(device: device, a: 0)
        mpscnn = activation
    }
    
    override public var type: String {
        return LayerModel.reluType
    }
}

/// 这一层 其实就是卷积核为 1 * 1的 ConvolutionLayer
@available(iOS 10.0, *)
class PointwiseConvolutionLayer: ConvolutionLayer {
    
    override init(device: MTLDevice,
                  config: LayerModel) throws{
        try super.init(device: device, config: config)
    }
    
    override var type: String{
        return LayerModel.pointWiseType
    }
}



