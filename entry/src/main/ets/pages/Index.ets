import { fileIo } from '@kit.CoreFileKit';
import { picker } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import { CompressionService } from '../service/CompressionService';
import { CompressionResult, DecompressionResult, CompressWithDataResult } from '../core/Types';

// Color palette
const PRIMARY = '#0A59F7';
const PRIMARY_LIGHT = '#EDF4FF';
const SUCCESS = '#64BB5C';
const SUCCESS_BG = '#F0FAF0';
const ERROR = '#E84026';
const ERROR_BG = '#FFF0ED';
const TEXT_PRIMARY = '#182431';
const TEXT_SECONDARY = '#99182431';
const TEXT_HINT = '#66182431';
const CARD_BG = '#FFFFFF';
const PAGE_BG = '#F1F3F5';
const DIVIDER = '#33000000';

@Entry
@Component
struct Index {
  private service: CompressionService = new CompressionService();

  @State currentTab: number = 0;

  // --- Compress state ---
  @State inputText: string = '';
  @State compressStatus: string = '';
  @State compressIsError: boolean = false;
  @State compressOriginalSize: number = 0;
  @State compressedSize: number = 0;
  @State compressRatio: number = 0;
  @State compressTime: number = 0;
  @State isCompressing: boolean = false;
  @State hasArchiveData: boolean = false;
  private lastArchiveData: Uint8Array = new Uint8Array(0);

  // --- Decompress state ---
  @State decompressStatus: string = '';
  @State decompressIsError: boolean = false;
  @State decompressOriginalSize: number = 0;
  @State decompressCompressedSize: number = 0;
  @State decompressTime: number = 0;
  @State decompressedText: string = '';
  @State isDecompressing: boolean = false;
  @State verifyStatus: string = '';
  @State verifyIsOk: boolean = false;

  // --- Info ---
  @State backendInfo: string = '';
  @State tokenizerInfo: string = '';

  aboutToAppear(): void {
    this.backendInfo = this.service.getBackendDisplayName();
    this.tokenizerInfo = this.service.getTokenizerDisplayName();
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) {
      return `${bytes} B`;
    } else if (bytes < 1048576) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / 1048576).toFixed(2)} MB`;
  }

  private compressInputText(): void {
    if (this.inputText.length === 0) {
      this.compressStatus = 'Please enter text first';
      this.compressIsError = true;
      return;
    }
    this.isCompressing = true;
    this.compressStatus = '';
    this.compressIsError = false;

    const output = this.service.compressTextToBytes(this.inputText);
    const r: CompressionResult = output.result;
    this.lastArchiveData = output.archiveData;
    this.hasArchiveData = output.archiveData.length > 0;

    if (r.success) {
      this.compressStatus = 'Compression complete';
      this.compressIsError = false;
      this.compressOriginalSize = r.originalSize;
      this.compressedSize = r.compressedSize;
      this.compressRatio = r.ratio;
      this.compressTime = r.timeMs;
    } else {
      this.compressStatus = r.error;
      this.compressIsError = true;
    }
    this.isCompressing = false;
  }

  private async selectAndCompressFile(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const documentPicker = new picker.DocumentViewPicker(context);
      const selectOptions = new picker.DocumentSelectOptions();
      const uris: string[] = await documentPicker.select(selectOptions);
      if (uris.length === 0) {
        return;
      }
      const uri: string = uris[0];
      this.isCompressing = true;
      this.compressStatus = '';
      this.compressIsError = false;

      const file = fileIo.openSync(uri, fileIo.OpenMode.READ_ONLY);
      const stat = fileIo.statSync(file.fd);
      const buf = new ArrayBuffer(stat.size);
      fileIo.readSync(file.fd, buf);
      fileIo.closeSync(file.fd);

      const output = this.service.compressBytes(new Uint8Array(buf));
      const r: CompressionResult = output.result;
      this.lastArchiveData = output.archiveData;
      this.hasArchiveData = output.archiveData.length > 0;

      if (r.success) {
        this.compressStatus = 'File compressed';
        this.compressIsError = false;
        this.compressOriginalSize = r.originalSize;
        this.compressedSize = r.compressedSize;
        this.compressRatio = r.ratio;
        this.compressTime = r.timeMs;
      } else {
        this.compressStatus = r.error;
        this.compressIsError = true;
      }
    } catch (e) {
      this.compressStatus = `${(e as Error).message || String(e)}`;
      this.compressIsError = true;
    }
    this.isCompressing = false;
  }

  private async saveArchive(): Promise<void> {
    if (!this.hasArchiveData) {
      return;
    }
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const filePath: string = `${context.filesDir}/output${this.service.getFileExtension()}`;
      const file = fileIo.openSync(filePath, fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY | fileIo.OpenMode.TRUNC);
      fileIo.writeSync(file.fd, this.lastArchiveData.buffer);
      fileIo.closeSync(file.fd);
      this.compressStatus = `Saved to sandbox`;
      this.compressIsError = false;
    } catch (e) {
      this.compressStatus = `Save failed: ${(e as Error).message || String(e)}`;
      this.compressIsError = true;
    }
  }

  private async selectAndDecompress(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const documentPicker = new picker.DocumentViewPicker(context);
      const selectOptions = new picker.DocumentSelectOptions();
      const uris: string[] = await documentPicker.select(selectOptions);
      if (uris.length === 0) {
        return;
      }
      this.isDecompressing = true;
      this.decompressStatus = '';
      this.decompressIsError = false;
      this.decompressedText = '';
      this.verifyStatus = '';

      const file = fileIo.openSync(uris[0], fileIo.OpenMode.READ_ONLY);
      const stat = fileIo.statSync(file.fd);
      const buf = new ArrayBuffer(stat.size);
      fileIo.readSync(file.fd, buf);
      fileIo.closeSync(file.fd);

      const r: DecompressionResult = this.service.decompressToText(new Uint8Array(buf));
      if (r.success) {
        this.decompressStatus = 'Decompression complete';
        this.decompressIsError = false;
        this.decompressedText = r.text;
        this.decompressOriginalSize = r.originalSize;
        this.decompressCompressedSize = r.compressedSize;
        this.decompressTime = r.timeMs;
      } else {
        this.decompressStatus = r.error;
        this.decompressIsError = true;
      }
    } catch (e) {
      this.decompressStatus = `${(e as Error).message || String(e)}`;
      this.decompressIsError = true;
    }
    this.isDecompressing = false;
  }

  private decompressLastArchive(): void {
    if (!this.hasArchiveData) {
      return;
    }
    this.isDecompressing = true;
    this.decompressStatus = '';
    this.decompressIsError = false;
    this.decompressedText = '';
    this.verifyStatus = '';

    const r: DecompressionResult = this.service.decompressToText(this.lastArchiveData);
    if (r.success) {
      this.decompressStatus = 'Decompression complete';
      this.decompressIsError = false;
      this.decompressedText = r.text;
      this.decompressOriginalSize = r.originalSize;
      this.decompressCompressedSize = r.compressedSize;
      this.decompressTime = r.timeMs;
      if (this.inputText.length > 0 && r.text === this.inputText) {
        this.verifyStatus = 'Lossless verified';
        this.verifyIsOk = true;
      } else if (this.inputText.length > 0) {
        this.verifyStatus = 'Mismatch detected';
        this.verifyIsOk = false;
      }
    } else {
      this.decompressStatus = r.error;
      this.decompressIsError = true;
    }
    this.isDecompressing = false;
  }

  @Builder
  MetricItem(label: string, value: string) {
    Column() {
      Text(value)
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor(TEXT_PRIMARY)
      Text(label)
        .fontSize(11)
        .fontColor(TEXT_HINT)
        .margin({ top: 2 })
    }
    .layoutWeight(1)
    .alignItems(HorizontalAlign.Center)
  }

  @Builder
  StatusBadge(text: string, isError: boolean) {
    Text(text)
      .fontSize(13)
      .fontColor(isError ? ERROR : SUCCESS)
      .backgroundColor(isError ? ERROR_BG : SUCCESS_BG)
      .borderRadius(12)
      .padding({ left: 12, right: 12, top: 6, bottom: 6 })
      .maxLines(2)
      .textOverflow({ overflow: TextOverflow.Ellipsis })
  }

  @Builder
  SectionCard() {
    Column()
      .width('100%')
      .backgroundColor(CARD_BG)
      .borderRadius(16)
      .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
  }

  @Builder
  CompressPage() {
    Column() {
      // Text input card
      Column() {
        Text('Input Text')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor(TEXT_SECONDARY)
          .margin({ bottom: 8 })

        TextArea({ placeholder: 'Paste or type text to compress...', text: this.inputText })
          .height(140)
          .width('100%')
          .backgroundColor('#F7F8FA')
          .borderRadius(12)
          .fontSize(14)
          .onChange((value: string) => {
            this.inputText = value;
          })
      }
      .width('100%')
      .padding(16)
      .backgroundColor(CARD_BG)
      .borderRadius(16)
      .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })

      // Action buttons
      Row() {
        Button('Compress Text')
          .type(ButtonType.Capsule)
          .backgroundColor(PRIMARY)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .layoutWeight(1)
          .onClick(() => {
            this.compressInputText();
          })
          .enabled(!this.isCompressing)

        Button('From File')
          .type(ButtonType.Capsule)
          .backgroundColor('#FFFFFF')
          .fontColor(PRIMARY)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .layoutWeight(1)
          .borderWidth(1)
          .borderColor(PRIMARY)
          .margin({ left: 12 })
          .onClick(() => {
            this.selectAndCompressFile();
          })
          .enabled(!this.isCompressing)
      }
      .width('100%')
      .margin({ top: 16 })

      // Status
      if (this.compressStatus.length > 0) {
        Row() {
          this.StatusBadge(this.compressStatus, this.compressIsError)
        }
        .width('100%')
        .justifyContent(FlexAlign.Start)
        .margin({ top: 12 })
      }

      // Metrics card
      if (this.compressOriginalSize > 0 && !this.compressIsError) {
        Column() {
          Row() {
            this.MetricItem('Original', this.formatBytes(this.compressOriginalSize))
            Column()
              .width(1)
              .height(32)
              .backgroundColor(DIVIDER)
            this.MetricItem('Compressed', this.formatBytes(this.compressedSize))
            Column()
              .width(1)
              .height(32)
              .backgroundColor(DIVIDER)
            this.MetricItem('Ratio', `${(this.compressRatio * 100).toFixed(1)}%`)
            Column()
              .width(1)
              .height(32)
              .backgroundColor(DIVIDER)
            this.MetricItem('Time', `${this.compressTime}ms`)
          }
          .width('100%')
          .justifyContent(FlexAlign.SpaceEvenly)
          .padding({ top: 16, bottom: 16 })
        }
        .width('100%')
        .backgroundColor(CARD_BG)
        .borderRadius(16)
        .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
        .margin({ top: 16 })
      }

      // Post-compress actions
      if (this.hasArchiveData) {
        Row() {
          Button('Save .lxf')
            .type(ButtonType.Capsule)
            .backgroundColor('#F7F8FA')
            .fontColor(TEXT_PRIMARY)
            .fontSize(13)
            .height(36)
            .layoutWeight(1)
            .onClick(() => {
              this.saveArchive();
            })

          Button('Verify Round-trip')
            .type(ButtonType.Capsule)
            .backgroundColor(PRIMARY_LIGHT)
            .fontColor(PRIMARY)
            .fontSize(13)
            .height(36)
            .layoutWeight(1)
            .margin({ left: 12 })
            .onClick(() => {
              this.decompressLastArchive();
              this.currentTab = 1;
            })
        }
        .width('100%')
        .margin({ top: 12 })
      }
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 24 })
  }

  @Builder
  DecompressPage() {
    Column() {
      // Open file card
      Column() {
        Text('Select an .lxf archive to decompress')
          .fontSize(14)
          .fontColor(TEXT_SECONDARY)
          .margin({ bottom: 16 })

        Button('Open Archive File')
          .type(ButtonType.Capsule)
          .backgroundColor(PRIMARY)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(44)
          .width('100%')
          .onClick(() => {
            this.selectAndDecompress();
          })
          .enabled(!this.isDecompressing)

        if (this.hasArchiveData) {
          Button('Decompress Last Compressed')
            .type(ButtonType.Capsule)
            .backgroundColor('#F7F8FA')
            .fontColor(TEXT_PRIMARY)
            .fontSize(13)
            .height(40)
            .width('100%')
            .margin({ top: 10 })
            .onClick(() => {
              this.decompressLastArchive();
            })
            .enabled(!this.isDecompressing)
        }
      }
      .width('100%')
      .padding(16)
      .backgroundColor(CARD_BG)
      .borderRadius(16)
      .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })

      // Status
      if (this.decompressStatus.length > 0) {
        Row() {
          this.StatusBadge(this.decompressStatus, this.decompressIsError)

          if (this.verifyStatus.length > 0) {
            Text(this.verifyStatus)
              .fontSize(12)
              .fontColor(this.verifyIsOk ? SUCCESS : ERROR)
              .fontWeight(FontWeight.Medium)
              .backgroundColor(this.verifyIsOk ? SUCCESS_BG : ERROR_BG)
              .borderRadius(10)
              .padding({ left: 10, right: 10, top: 4, bottom: 4 })
              .margin({ left: 8 })
          }
        }
        .width('100%')
        .justifyContent(FlexAlign.Start)
        .margin({ top: 12 })
      }

      // Metrics
      if (this.decompressOriginalSize > 0 && !this.decompressIsError) {
        Column() {
          Row() {
            this.MetricItem('Original', this.formatBytes(this.decompressOriginalSize))
            Column()
              .width(1)
              .height(32)
              .backgroundColor(DIVIDER)
            this.MetricItem('Archive', this.formatBytes(this.decompressCompressedSize))
            Column()
              .width(1)
              .height(32)
              .backgroundColor(DIVIDER)
            this.MetricItem('Time', `${this.decompressTime}ms`)
          }
          .width('100%')
          .justifyContent(FlexAlign.SpaceEvenly)
          .padding({ top: 16, bottom: 16 })
        }
        .width('100%')
        .backgroundColor(CARD_BG)
        .borderRadius(16)
        .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
        .margin({ top: 16 })
      }

      // Output text
      if (this.decompressedText.length > 0) {
        Column() {
          Text('Decompressed Output')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor(TEXT_SECONDARY)
            .margin({ bottom: 8 })

          Scroll() {
            Text(this.decompressedText)
              .fontSize(13)
              .fontColor(TEXT_PRIMARY)
              .width('100%')
          }
          .width('100%')
          .height(200)
          .backgroundColor('#F7F8FA')
          .borderRadius(12)
          .padding(12)
          .scrollBar(BarState.Auto)
        }
        .width('100%')
        .padding(16)
        .backgroundColor(CARD_BG)
        .borderRadius(16)
        .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
        .margin({ top: 16 })
        .alignItems(HorizontalAlign.Start)
      }
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 24 })
  }

  build() {
    Column() {
      // === Top bar ===
      Column() {
        Row() {
          Column() {
            Text('LexiFold')
              .fontSize(22)
              .fontWeight(FontWeight.Bold)
              .fontColor(TEXT_PRIMARY)
            Text('Predictive Text Compression')
              .fontSize(12)
              .fontColor(TEXT_HINT)
              .margin({ top: 2 })
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)

          Column() {
            Text('CPU')
              .fontSize(11)
              .fontColor(PRIMARY)
              .fontWeight(FontWeight.Bold)
            Text('Byte-256')
              .fontSize(10)
              .fontColor(TEXT_HINT)
              .margin({ top: 1 })
          }
          .alignItems(HorizontalAlign.End)
          .padding({ left: 12, right: 4, top: 4, bottom: 4 })
          .backgroundColor(PRIMARY_LIGHT)
          .borderRadius(8)
        }
        .width('100%')
        .padding({ left: 20, right: 20, top: 12, bottom: 12 })
      }
      .width('100%')
      .backgroundColor(CARD_BG)

      // === Tabs ===
      Tabs({ barPosition: BarPosition.Start, index: this.currentTab }) {
        TabContent() {
          Scroll() {
            this.CompressPage()
          }
          .width('100%')
          .height('100%')
          .scrollBar(BarState.Off)
        }
        .tabBar(this.TabBarItem('Compress', 0))

        TabContent() {
          Scroll() {
            this.DecompressPage()
          }
          .width('100%')
          .height('100%')
          .scrollBar(BarState.Off)
        }
        .tabBar(this.TabBarItem('Decompress', 1))
      }
      .barMode(BarMode.Fixed)
      .barWidth('100%')
      .barHeight(48)
      .onChange((index: number) => {
        this.currentTab = index;
      })
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(PAGE_BG)
  }

  @Builder
  TabBarItem(title: string, index: number) {
    Column() {
      Text(title)
        .fontSize(15)
        .fontWeight(this.currentTab === index ? FontWeight.Bold : FontWeight.Normal)
        .fontColor(this.currentTab === index ? PRIMARY : TEXT_HINT)
      Column()
        .width(24)
        .height(3)
        .borderRadius(2)
        .backgroundColor(this.currentTab === index ? PRIMARY : 'transparent')
        .margin({ top: 6 })
    }
    .justifyContent(FlexAlign.Center)
    .height('100%')
    .width('100%')
  }
}
